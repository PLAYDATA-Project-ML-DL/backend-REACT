"""
import logging.config
from logstash import TCPLogstashHandler

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'logstash': {
            'level': 'DEBUG',
            'class': 'logstash.TCPLogstashHandler',
            'host': 'localhost',
            'port': 5000,
            'version': 1,
            'message_type': 'django',
            'fqdn': False,
            'tags': ['django'],
        },
    },
    'loggers': {
        'django': {
            'handlers': ['logstash'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

logging.config.dictConfig(LOGGING)"""