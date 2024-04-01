# VisionTests
VisionTests

## API Key info
add file 'api_key.py' with the content 

```
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("add appi key here"),
)
```

## Enviroment 
a python 3.11 enviroment was configured 
```
.venv/Scripts/python.exe -m pip install pyrealsense2 opencv-python dlib
```