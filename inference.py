import requests
import json
import pprint
import threading


def make_request(url: str, post: bool = False, data=None):
    if post:
        url_obj = requests.post(url, data=data)
    else:
        url_obj = requests.get(url, data=data)
    text = url_obj.text
    data = json.loads(text)
    return data


def inference(text: str, pp):
    model_name = "synthetic"
    version = "1"
    url = f"http://localhost:18080/predictions/{model_name}/{version}"
    result = make_request(url, False, {"data": text})
    result["text"] = text
    # result["softmax"] =
    pp.pprint(result)


def test(data, parallel=False):
    pp = pprint.PrettyPrinter(indent=4)
    for text in data:
        t = threading.Thread(target=inference, args=(text, pp))
        if parallel:
            t.start()
        else:
            t.run()


if __name__ == '__main__':
    data = [
        "137-02557628",
        "13702557628",
        "440602199110191812",
        "张帅",
    ]
    test(data)
