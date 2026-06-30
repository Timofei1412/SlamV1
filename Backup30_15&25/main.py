import logging
from datetime import datetime
from сommunicate import ESPCommunication

def main():
    timestamp = datetime.now().strftime('%d_%H-%M-%S')
    log_file = f'Output/Logs/app_{timestamp}.log'

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        # datefmt='%H:%M:%S.%f',
        format='%(asctime)s | %(levelname)-8s | %(filename)s | %(message)s',
        level=logging.INFO,
        encoding='utf-8',
        # backupCount=3,
        )

    ESP = ESPCommunication(debug=False)
    ESP.sendMotionCommand([0, 0, 0, 0], [90, 90, 90, 90])
    ESP.close()

if __name__ == "__main__":
    main()