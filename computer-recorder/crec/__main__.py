from dotenv import load_dotenv
load_dotenv()

import asyncio
from .cli import main
 
if __name__ == '__main__':
    asyncio.run(main()) 