//! GPU utilization from the `IOAccelerator` registry entries. These are public
//! `IOKit` properties, readable inside the App Sandbox without opening a
//! driver.

use std::ffi::{c_char, c_void};

use core_foundation::{
    base::{CFType, TCFType},
    dictionary::CFDictionary,
    number::CFNumber,
    string::CFString,
};

/// An `io_object_t`, `io_service_t` or `io_iterator_t`.
type IoObject = u32;

/// `kIOMainPortDefault`.
const MAIN_PORT_DEFAULT: u32 = 0;
const KERN_SUCCESS: i32 = 0;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingServices(
        main_port: u32,
        matching: *mut c_void,
        existing: *mut IoObject,
    ) -> i32;
    fn IOIteratorNext(iterator: IoObject) -> IoObject;
    fn IORegistryEntryCreateCFProperty(
        entry: IoObject,
        key: *const c_void,
        allocator: *const c_void,
        options: u32,
    ) -> *const c_void;
    fn IOObjectRelease(object: IoObject) -> i32;
}

/// The busiest accelerator's `Device Utilization %`, from zero to one hundred,
/// or `None` when no accelerator reports it.
pub fn utilization_percent() -> Option<f64> {
    let statistics = CFString::from_static_string("PerformanceStatistics");
    let utilization = CFString::from_static_string("Device Utilization %");
    let mut iterator: IoObject = 0;
    // SAFETY: `IOServiceGetMatchingServices` takes ownership of the matching
    // dictionary, null included, and fills `iterator` on success.
    let status = unsafe {
        IOServiceGetMatchingServices(
            MAIN_PORT_DEFAULT,
            IOServiceMatching(c"IOAccelerator".as_ptr()),
            &raw mut iterator,
        )
    };
    if status != KERN_SUCCESS {
        return None;
    }
    let mut busiest: Option<f64> = None;
    loop {
        // SAFETY: `iterator` is a valid iterator until it is released below.
        let entry = unsafe { IOIteratorNext(iterator) };
        if entry == 0 {
            break;
        }
        if let Some(value) = entry_utilization(entry, &statistics, &utilization) {
            busiest = Some(busiest.map_or(value, |current| current.max(value)));
        }
        // SAFETY: `entry` came from `IOIteratorNext` and is released once.
        unsafe { IOObjectRelease(entry) };
    }
    // SAFETY: `iterator` came from `IOServiceGetMatchingServices`.
    unsafe { IOObjectRelease(iterator) };
    busiest
}

fn entry_utilization(entry: IoObject, statistics: &CFString, key: &CFString) -> Option<f64> {
    // SAFETY: `entry` is a live registry entry and `statistics` a CFString;
    // the property follows the create rule and is wrapped below.
    let property = unsafe {
        IORegistryEntryCreateCFProperty(entry, statistics.as_CFTypeRef(), std::ptr::null(), 0)
    };
    if property.is_null() {
        return None;
    }
    // SAFETY: a non-null property returned under the create rule.
    let property = unsafe { CFType::wrap_under_create_rule(property) };
    let statistics = property.downcast_into::<CFDictionary>()?;
    let value = statistics.find(key.as_CFTypeRef())?;
    // SAFETY: dictionary values are CF objects the dictionary keeps alive.
    let value = unsafe { CFType::wrap_under_get_rule(*value) };
    value
        .downcast::<CFNumber>()?
        .to_f64()
        .filter(|percent| percent.is_finite() && *percent >= 0.0)
        .map(|percent| percent.min(100.0))
}
