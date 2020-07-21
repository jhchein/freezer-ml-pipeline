import logging


def get_connection_device_id(data):
    # Check if ConnectionDeviceId is provided as a key value pair.
    connection_device_id = data.get("ConnectionDeviceId")
    if connection_device_id is not None:
        logging.info(f"ConnectionDeviceId: '{connection_device_id}'")
        return connection_device_id, False, None

    # If not connection_device_id key exists, get connection_device_id from the single events
    unique_connection_device_ids = list(
        set(
            [
                event.get("ConnectionDeviceId")
                for event in data.get("allevents")
                if event.get("ConnectionDeviceId") is not None
            ]
        )
    )

    if len(unique_connection_device_ids) > 1:
        error_message = (
            f"Multiple ConnectionDeviceIds found ({unique_connection_device_ids})."
        )
        logging.warning(error_message)
        return None, True, error_message
    if len(unique_connection_device_ids) < 1:
        error_message = "No ConnectionDeviceIds found."
        logging.warning(error_message)
        return None, True, error_message

    logging.info(f"ConnectionDeviceId: '{unique_connection_device_ids[0]}'")
    return unique_connection_device_ids[0], False, None


def create_response(
    prediction=None,
    connection_device_id=None,
    time_created_start="",
    time_created_end="",
    has_error=False,
    error_message=None,
):
    return {
        "result": prediction,
        "ConnectionDeviceId": connection_device_id,
        "timeCreatedStart": time_created_start,
        "timeCreatedEnd": time_created_end,
        "hasError": has_error,
        "errorMessage": error_message,
    }
