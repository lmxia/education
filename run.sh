curl -F 'file=@./tiger.jpeg' localhost:8000/image/process
curl -X POST -H 'content-type:application/json' -d '{"points":[{"x":"1","y":"0.9"},{"x":"2.1","y":"1.9"},{"x":"2.9","y":"2.87"}]}' 18.216.109.187:8000/line/process
git clone https://github.com/lmxia/education.git
git clone https://github.com/tensorlayer/tensorlayer.git
cd education && wget https://github.com/tensorlayer/pretrained-models/raw/master/models/vgg19.npy


pip install django djangorestframework
apt update && apt install python-tk

python manage.py runserver 0.0.0.0:8000
docker run -it -d --rm -p 8888:8888 -p 6006:6006 -p 8000:8000 -v $(pwd):/notebooks -e PASSWORD=12345 tensorlayer/tensorlayer:latest-gpu