### Secure split inference with ZK-SNARK based verification
This is a basic proof of concept for Split Inference that would enable clients to be able to obfuscate their inputs and run the first few lightweight layers of the target model on the client machine. Then using **gRPC**,
we can send **AES-GCM**-encrypted weights to the server for complete inference. The final response can be sent back to the client for usage. All through this process, the actual inputs and related metadata are never
revealed to the server.

But the question arises how the server can enforce constraints on inputs if it has no knowledge of them, and also how does the client know that the server has performed a complete inference and not some malicious action?
For this, we use lightweight ZK-Proofs. Written in **Noir** with Baretenberg backend, a basic infernece cycle is as follows:

## STEP 1:
Client gives input to model -> model runs first lightweight layers on client machine and encrypts the resulting weights
## step 2:
Client sends encrypted weights and a proof of valid inputs to the server with verification key using gRPC.
## step 3: 
The server recives the weights and verifies the proof before proceeding. If the porrf is valid, I,e the input is of an accpetbale field, the server runs the remaining and compuataiontally intesive part of inference
and sends back the ouput with another proof that the weights sent by the client were only used for inference.
## step 4: 
The client then recives the proof and the response, and is thus able to ascertain no malicious activity has taken place with the data.
This way the server and client are able to maintain anonymity of data to some extent and preventing model reverse enginerring or extracing data for malicious party.


