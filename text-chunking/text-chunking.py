def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    chunks = []
    pos = 0
    while pos < len(tokens) - overlap:
        chunks.append(tokens[pos:pos+chunk_size])
        pos += chunk_size - overlap
    return chunks