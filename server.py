import asyncio
import websockets
import time
# from demo import MonitorBabay


all_clients = []

# async def send_message(message:str):
#     for client in all_clients:
#         await client.send(message)


def get_all_clients():
    return all_clients

async def send_message(person_distance):
    for client in all_clients:
        print('about to send message', person_distance)
        await client.send(person_distance)

async def new_client_connected(client_socket,path):
    print("new client connected!")
    all_clients.append(client_socket)
    while True:
        new_message = await client_socket.recv()
        print("client sent:", new_message)
        await send_message(new_message)



async def start_server():
    print("server started!")
    await websockets.serve(new_client_connected,"localhost",12345)

   

if __name__ == "__main__":
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(start_server())
    event_loop.run_forever()


