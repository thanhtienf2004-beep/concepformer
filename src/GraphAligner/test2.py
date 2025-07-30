import h5py
with h5py.File('C:\\Users\\thanh\\ConceptFormer\\data\\artifacts\\BigGraphAlignment_v1\\TRExStar\\gpt2\\init\\embeddings_entity_0.v0.h5', 'r') as hf:
    print(hf['embeddings'].shape)