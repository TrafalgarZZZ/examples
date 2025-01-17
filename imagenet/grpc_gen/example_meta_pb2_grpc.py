# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpc_gen.example_meta_pb2 as example__meta__pb2


class DatasetServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.FetchExample = channel.unary_stream(
                '/indexer.DatasetService/FetchExample',
                request_serializer=example__meta__pb2.ExampleRequest.SerializeToString,
                response_deserializer=example__meta__pb2.ExampleMeta.FromString,
                )
        self.Register = channel.unary_unary(
                '/indexer.DatasetService/Register',
                request_serializer=example__meta__pb2.RegisterRequest.SerializeToString,
                response_deserializer=example__meta__pb2.RegisterResponse.FromString,
                )


class DatasetServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def FetchExample(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Register(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DatasetServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'FetchExample': grpc.unary_stream_rpc_method_handler(
                    servicer.FetchExample,
                    request_deserializer=example__meta__pb2.ExampleRequest.FromString,
                    response_serializer=example__meta__pb2.ExampleMeta.SerializeToString,
            ),
            'Register': grpc.unary_unary_rpc_method_handler(
                    servicer.Register,
                    request_deserializer=example__meta__pb2.RegisterRequest.FromString,
                    response_serializer=example__meta__pb2.RegisterResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'indexer.DatasetService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DatasetService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def FetchExample(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/indexer.DatasetService/FetchExample',
            example__meta__pb2.ExampleRequest.SerializeToString,
            example__meta__pb2.ExampleMeta.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Register(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/indexer.DatasetService/Register',
            example__meta__pb2.RegisterRequest.SerializeToString,
            example__meta__pb2.RegisterResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
