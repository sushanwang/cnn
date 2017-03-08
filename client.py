import grpc

import cnn_model_pb2_grpc
import cnn_model_pb2
import sys


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = cnn_model_pb2_grpc.CnnModelStub(channel)
    for query in sys.stdin:
        reply = stub.ScoreCnnModel(cnn_model_pb2.QueryRequest(query=query))
        print(reply.score)

if __name__ == '__main__':
    run()