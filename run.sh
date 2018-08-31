curl -F 'file=@./tiger.jpeg' localhost:8000/image/process
git clone https://github.com/lmxia/education.git
git clone https://github.com/tensorlayer/tensorlayer.git

pip install django djangorestframework
docker run -it -d --rm -p 8888:8888 -p 6006:6006 -p 8000:8000 -v $(pwd):/notebooks -e PASSWORD=12345 tensorlayer/tensorlayer:latest-gpu