import datetime
import pandas as pd
import io
import os


class Logger:

    def __init__(self, filePath=None):
        self.filePath = filePath

        self.file = None
        if self.filePath:
            s = io.StringIO()

            df = self._log_dataframe()
            df.to_csv(s, header=True, index=False)

            if not os.path.isfile(self.filePath):
                open(self.filePath, 'w+').close()

            correct_header = s.getvalue()
            f = open(self.filePath, 'r')
            header = f.readline()
            f.close()
            if header != correct_header:
                f = open(self.filePath, 'w+')
                f.write(correct_header)
                f.close()

            self.file = open(self.filePath, 'a+')

    def _log_dataframe(self, datetimes=[], levels=[], types=[], descriptions=[], metas=[]):
        return pd.DataFrame({'datetime': datetimes, 'level': levels, 'type': types, 'description': descriptions, 'meta': metas})

    def line(self, level, type, description, meta):
        if self.file is None: return

        s = io.StringIO()

        df = self._log_dataframe([str(datetime.datetime.now())], [level], [type], [description], [meta])
        df.to_csv(s, header=False, index=False)

        self.file.write(s.getvalue())
        self.file.flush()


    def d(self, type, description, meta=''):
        self.line('d', type, description, meta)

    def t(self, type, description, meta=''):
        self.line('t', type, description, meta)

    def l(self, type, description, meta=''):
        self.line('l', type, description, meta)

    def i(self, type, description, meta=''):
        self.line('i', type, description, meta)

    def w(self, type, description, meta=''):
        self.line('w', type, description, meta)

    def e(self, type, description, meta=''):
        self.line('e', type, description, meta)
