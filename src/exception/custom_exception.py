import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail:sys):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is None:
            # Raised outside an active except block; there is no traceback to read and
            # dereferencing it here would mask the original error with an AttributeError.
            return f"{error_message}"

        # Walk to the deepest frame so the reported location is where the error actually
        # occurred, not the outermost handler that re-raised it.
        while exc_tb.tb_next is not None:
            exc_tb = exc_tb.tb_next
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error in {file_name}, line {line_number} : {error_message}"

    def __str__(self):
        return self.error_message
