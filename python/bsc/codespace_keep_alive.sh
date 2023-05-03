#!/bin/bash

# Replace with your desired interval in seconds
INTERVAL=60

# Replace with your desired number of requests
NUM_REQUESTS=10

# Loop through the number of requests
for ((i=1;i<=NUM_REQUESTS;i++))
do
    # Send a GET request to the codespace URL
    RESPONSE1=$(curl -s -w "%{http_code}" 127.0.0.1:3000)
    RESPONSE2=$(curl -s -w "%{http_code}" 127.0.0.1:3001)
    RESPONSE3=$(curl -s -w "%{http_code}" 127.0.0.1:6006)
    RESPONSE4=$(curl -s -w "%{http_code}" 127.0.0.1:32781)
    RESPONSE5=$(curl -s -w "%{http_code}" 127.0.0.1:35289)


    # Print the status code and the content
    echo "Request $i: ${RESPONSE1: -3}"
    echo "${RESPONSE1::-3}"

    echo "Request $i: ${RESPONSE1: -3}"
    echo "${RESPONSE1::-3}"

    echo "Request $i: ${RESPONSE3: -3}"
    echo "${RESPONSE3::-3}"

    echo "Request $i: ${RESPONSE4: -3}"
    echo "${RESPONSE4::-3}"

    echo "Request $i: ${RESPONSE5: -3}"
    echo "${RESPONSE5::-3}"

    # Wait for the interval before sending the next request
    sleep $INTERVAL
done