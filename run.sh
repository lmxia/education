curl -F 'file=@./tiger.jpeg' localhost:8000/image/process
curl -F 'content_file=@./content_2.png' -F 'style_file=@./style_2.png' 18.191.56.202:8000/style/transfer
curl -X POST -H 'content-type:application/json' -d '{"points":[{"x":"1","y":"0.9"},{"x":"2.1","y":"1.9"},{"x":"2.9","y":"2.87"}]}' 18.216.109.187:8000/line/process
git clone https://github.com/lmxia/education.git
git clone https://github.com/tensorlayer/tensorlayer.git
cd education && wget https://github.com/tensorlayer/pretrained-models/raw/master/models/vgg19.npy && wget https://github.com/tensorlayer/pretrained-models/raw/master/models/style_transfer_models_and_examples/pretrained_vgg19_encoder_model.npz \
wget https://github.com/tensorlayer/pretrained-models/raw/master/models/style_transfer_models_and_examples/pretrained_vgg19_decoder_model.npz
docker run -it -d --rm -p 8888:8888 -p 6006:6006 -p 8000:8000 -v $(pwd):/notebooks -e PASSWORD=12345 tensorlayer/tensorlayer:latest-gpu

pip install django djangorestframework
apt update && apt install python-tk

python manage.py runserver 0.0.0.0:8000
