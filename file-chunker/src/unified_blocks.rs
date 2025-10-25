/// A unified representation of sequential text blocks from various readers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifiedBlock {
    pub text: String,
}

impl UnifiedBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

