import grpc

import cnn_model_pb2_grpc
import cnn_model_pb2
import sys


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = cnn_model_pb2_grpc.CnnModelStub(channel)
    for query in sys.stdin:
        result = cnn_model_pb2.QueryRequest(query=query)
        replies = stub.ScoreCnnModel(result)
        for reply in replies:
            print("the app name is %s with prob %.6f" % (reply.scorelist.query, reply.scorelist.prob))


if __name__ == '__main__':
    run()