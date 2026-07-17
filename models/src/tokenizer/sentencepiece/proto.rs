use crate::{
    error::{ModelsError, Result},
    tokenizer::sentencepiece::{PieceKind, SentencePieceModel, SpPiece},
};

#[derive(Debug)]
struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

pub fn parse(bytes: &[u8]) -> Result<SentencePieceModel> {
    let mut reader = Reader::new(bytes);
    let mut pieces = Vec::new();
    let mut model_type = None;
    while let Some((field, wire)) = reader.next_key()? {
        match (field, wire) {
            (1, 2) => pieces.push(parse_piece(reader.bytes()?)?),
            (2, 2) => model_type = parse_trainer(reader.bytes()?)?,
            _ => reader.skip(wire)?,
        }
    }
    if pieces.is_empty() {
        return Err(ModelsError::InvalidConfig("SentencePiece model has no pieces".into()));
    }
    Ok(SentencePieceModel { pieces, model_type })
}

fn parse_piece(bytes: &[u8]) -> Result<SpPiece> {
    let mut reader = Reader::new(bytes);
    let mut piece = None;
    let mut score = 0.0;
    let mut kind = PieceKind::Normal;
    while let Some((field, wire)) = reader.next_key()? {
        match (field, wire) {
            (1, 2) => piece = Some(reader.string()?.to_owned()),
            (2, 5) => score = reader.float32()?,
            (3, 0) => kind = PieceKind::from_u64(reader.varint()?),
            _ => reader.skip(wire)?,
        }
    }
    let piece = piece
        .ok_or_else(|| ModelsError::InvalidConfig("SentencePiece piece is missing text".into()))?;
    Ok(SpPiece { piece, score, kind })
}

fn parse_trainer(bytes: &[u8]) -> Result<Option<u32>> {
    let mut reader = Reader::new(bytes);
    let mut model_type = None;
    while let Some((field, wire)) = reader.next_key()? {
        match (field, wire) {
            (3, 0) => model_type = Some(u32::try_from(reader.varint()?)?),
            _ => reader.skip(wire)?,
        }
    }
    Ok(model_type)
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn next_key(&mut self) -> Result<Option<(u32, u8)>> {
        if self.cursor == self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        Ok(Some((u32::try_from(key >> 3)?, u8::try_from(key & 0x07)?)))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        let mut shift = 0_u32;
        loop {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(ModelsError::InvalidConfig("protobuf varint is too long".into()));
            }
        }
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.varint()?)?;
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| ModelsError::InvalidConfig("protobuf length overflows usize".into()))?;
        if end > self.bytes.len() {
            return Err(ModelsError::InvalidConfig("protobuf length exceeds input".into()));
        }
        let out = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(out)
    }

    fn string(&mut self) -> Result<&'a str> {
        Ok(std::str::from_utf8(self.bytes()?)?)
    }

    fn float32(&mut self) -> Result<f32> {
        let bytes = self.take::<4>()?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                let _value = self.varint()?;
            },
            1 => {
                let _bytes = self.take::<8>()?;
            },
            2 => {
                let _bytes = self.bytes()?;
            },
            5 => {
                let _bytes = self.take::<4>()?;
            },
            _ => {
                return Err(ModelsError::InvalidConfig(format!(
                    "unsupported protobuf wire {wire}"
                )));
            },
        }
        Ok(())
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.cursor.checked_add(N).ok_or_else(|| {
            ModelsError::InvalidConfig("protobuf fixed field overflows usize".into())
        })?;
        if end > self.bytes.len() {
            return Err(ModelsError::InvalidConfig("protobuf fixed field exceeds input".into()));
        }
        let mut out = [0; N];
        out.copy_from_slice(&self.bytes[self.cursor..end]);
        self.cursor = end;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8> {
        let byte =
            self.bytes.get(self.cursor).copied().ok_or_else(|| {
                ModelsError::InvalidConfig("unexpected end of protobuf input".into())
            })?;
        self.cursor += 1;
        Ok(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sentencepiece_piece_and_trainer_type() -> Result<()> {
        let bytes = [
            0x0a, 0x0e, 0x0a, 0x05, b'<', b'u', b'n', b'k', b'>', 0x15, 0, 0, 0, 0, 0x18, 0x02,
            0x12, 0x02, 0x18, 0x02,
        ];
        let model = parse(&bytes)?;

        assert_eq!(model.pieces.len(), 1);
        assert_eq!(model.pieces[0].piece, "<unk>");
        assert_eq!(model.pieces[0].kind, PieceKind::Unknown);
        assert_eq!(model.model_type, Some(2));
        Ok(())
    }
}
