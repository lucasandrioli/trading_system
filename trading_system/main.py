#!/usr/bin/env python
"""
Trading System - Entry Point
------------------------------------------
Este arquivo serve como ponto de entrada principal para a aplicação.
Execute este arquivo para iniciar o servidor da aplicação.

Uso:
    python app.py

Variáveis de ambiente:
    FLASK_ENV: Defina como 'development' para modo debug
    PORT: Porta do servidor (padrão: 5001)
    DATA_FOLDER: Caminho para o diretório de dados (padrão: 'data')
    DEBUG: Defina como 'True' para saída de debug (padrão: False)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import argparse

# Configuração do parser de argumentos
parser = argparse.ArgumentParser(description='Trading System Application')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)),
                   help='Porta para executar o servidor')
parser.add_argument('--debug', action='store_true', default=os.environ.get('DEBUG', 'False').lower() == 'true',
                   help='Executar em modo debug')
parser.add_argument('--data-folder', type=str, default=os.environ.get('DATA_FOLDER', 'data'),
                   help='Caminho para pasta de dados')
args = parser.parse_args()

# Garantir que o diretório de dados existe
os.makedirs(args.data_folder, exist_ok=True)

# Configurar logs
log_folder = os.path.join(args.data_folder, 'logs')
os.makedirs(log_folder, exist_ok=True)

log_level = logging.DEBUG if args.debug else logging.INFO
log_file = os.path.join(log_folder, 'trading_system.log')

# Configurar logger principal
logger = logging.getLogger()
logger.setLevel(log_level)

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# Handler para arquivo com rotação
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(log_level)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Importar e criar a aplicação Flask
from trading_system.web import create_app

# Configuração
config = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev_key_substituir_em_producao'),
    'DATA_FOLDER': args.data_folder,
    'CACHE_EXPIRY': int(os.environ.get('CACHE_EXPIRY', 3600)),
    'POLYGON_API_KEY': os.environ.get('POLYGON_API_KEY', ''),
    'DEBUG': args.debug,
    'TESTING': False,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max upload
}

# Criar a aplicação
app = create_app(config)

# Executar a aplicação
if __name__ == '__main__':
    logger.info(f"Iniciando Trading System na porta {args.port} com debug={args.debug}")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)