class SSMLException(Exception):
    def __init__(self, text='Invalid SSML'):
        super().__init__(text)
