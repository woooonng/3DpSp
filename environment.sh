cd dataset_preprocessing/ffhq/
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
git clone https://github.com/elliottwu/unsup3d.git
cd ../..

cd dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r insightface/recognition/arcface_torch ./models/
rm -rf insightface
cd ../../..

# this is for 3DFaceRecon
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
cd ..
rm -rf nvdiffrast

# this is for confidence map
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/load_textures_cuda.cpp
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/create_texture_image_cuda.cpp
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/rasterize_cuda.cpp
pip install .
cd ..
rm -rf neural_renderer