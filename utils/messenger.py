import zmq
import json


class Messenger:
    def __init__(self, port: str = "5511") -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*{port}")

    def publish(self, data: list) -> None:
        assert isinstance(data, list)
        message = json.dumps(data)
        self.socket.send_string(message)