# python
import contextlib
import threading
import time

# 3rdparty
import uvicorn


class Server(uvicorn.Server):
    """Обертка над uvicorn.Server, не блокирующая основной поток"""

    @contextlib.contextmanager
    def run_in_thread(self):
        """Метод для запуска сервиса в потоке"""
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started and thread.is_alive():
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()
