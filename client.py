# client.py
import zmq
import numpy as np
import json

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5511")

    for request_nbr in range(10):
        # Create a vector to send
        # vec = np.array([request_nbr, request_nbr + 1, request_nbr + 2], dtype=np.float64)
        # print(f"Sending vector: {vec}")
        data = "hello"
        print("sending hello")

        # Send the vector
        # socket.send(vec.tobytes())
        msg = json.dumps(data)
        socket.send_string(msg)

        # Get the reply
        message = socket.recv()
        # reply_vec = np.frombuffer(message, dtype=np.float64)
        print(f"Received vector: {message}")

if __name__ == "__main__":
    main()
