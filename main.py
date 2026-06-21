import logging
from datetime import datetime


def main():
    timestamp = datetime.now().strftime('%d_%H-%M-%S')
    log_file = f'Output/Logs/app_{timestamp}.log'

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        # datefmt='%H:%M:%S.%f',
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        level=logging.DEBUG,
        encoding='utf-8',
        # backupCount=3,  
        )

    logging.info('Приложение запущено')
    logging.debug('Отладочная информация')

if __name__ == "__main__":
    main()