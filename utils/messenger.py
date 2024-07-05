import zmq
import json
import numpy as np


class Messenger:
    def __init__(self, port: str = "5511") -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:" + port)

    def publish(self, data: list) -> None:
        assert isinstance(data, list)
        message = json.dumps(data)
        self.socket.send_string(message)

class Client:
    def __init__(self, port: str = "5511") -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5511")

    def send_request(self, data: list) -> None:
        # assert isinstance(data, list)
        message = json.dumps(data)
        self.socket.send_string(message)
        # self.socket.send(data)

    def get_reply(self):
        message = self.socket.recv()
        message = np.frombuffer(message, dtype=np.float64)
        return message