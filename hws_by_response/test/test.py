from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-base', cache_folder='./models_cache')
model.max_seq_length = 512
text = '안녕하세요' * 100  # 긴 텍스트
print('Encoding...')
vec = model.encode([text], normalize_embeddings=True)
print(f'Done: {vec.shape}')