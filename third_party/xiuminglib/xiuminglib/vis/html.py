from os.path import dirname

from ..os import makedirs


class HTML():
    """HTML Builder.

    Attributes:
        head (str)
        body (str)
        tail (str): Concatenating the three gives the complete file contents,
            as a string.
        children (dict): Child elements, such as a table.
    """
    def __init__(
            self, title="Results", bgcolor='black', text_font='roboto',
            text_color='white'):
        """
        Args:
            title (str, optional): Page title.
            bgcolor (str, optional): Background color. Supports at least color
                names (like, ``'black'``) and Hex colors (like ``'#FFFFFF'``).
            text_font (str, optional): Supported values include ``'arial'``,
                etc.
            text_color (str, optional): Text color.
        """
        # Start string
        self.head = '''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
</head>
<body bgcolor="{bgcolor}">
<font face="{text_font}" color="{text_color}">
'''.format(title=title,
           bgcolor=bgcolor,
           text_font=text_font,
           text_color=text_color)
        # Contents, to add
        self.body = ''
        # End string
        self.tail = '''
</font>
</body>
</html>
'''
        self.children = {}

    def add_header(self, text, level=1):
        self.body += '''
    <h{level}>{text}</h{level}>
'''.format(level=level,
           text=text)

    def add_table(self, name=None, header=None, width='100%', border=6):
        """Adds a table to the HTML's children.

        Args:
            name (str, optional): Table name to enable easy access:
                ``self.children[name]``.
            header (list(str), optional): Table header.
            width (str, optional): Table width.
            border (int, optional): Border width.

        Returns:
            :class:`xiuminglib.vis.html.Table`: Table added.
        """
        table = Table(header=header, width=width, border=border)
        if name is None:
            i = 0
            while True:
                name = 'table%d' % i
                if name not in self.children:
                    break
                i += 1
        else:
            assert name not in self.children, \
                "A child already took the name '%s'" % name
        self.children[name] = table
        return table

    def save(self, index_file):
        """Saves the generated HTML string to the index file.

        Once called, this method also calls all children's :func:`close`,
        so that everything (e.g., a table) is properly closed.

        Args:
            index_file (str): Path to the generated index.html.

        Writes
            - An HTML index file.
        """
        if not index_file.endswith('.html'):
            index_file += '.html'
        makedirs(dirname(index_file))
        for _, child in self.children.items():
            self.body += child.close()
        html_str = self.head + self.body + self.tail
        with open(index_file, 'w') as h:
            h.write(html_str)


class Table():
    """HTML Table.

    Attributes:
        head (str)
        body (str)
        tail (str): Concatenating the three gives the complete file contents,
            as a string.
        td (str): ``td`` start string that specifies a uniform style.
    """
    def __init__(self, header=None, width='100%', border=6):
        """
        Args:
            header (list(str), optional): Table headers.
            width (str, optional): Table width.
            border (int, optional): Border width.
        """
        self.head = '''
    <table style="width:{width}" border="{border}">
'''.format(width=width, border=border)
        self.body = '''
        <tr>
'''
        if header is not None:
            for col_header in header:
                self.body += '''
            <th><p>{text}</p></th>
'''.format(text=col_header)
        self.body += '''
        </tr>
'''
        self.tail = '''
    </table>
'''
        self.td = '<td align="center" valign="middle">'

    def close(self):
        """Closes the table.

        Returns:
            str: The generated table as a string.
        """
        return self.head + self.body + self.tail

    def add_row(self, media, types, captions=None, media_width=256):
        """Adds a row to the table.

        Args:
            media (list(str)): Paths to media, or the text to display itself.
            types (list(str)): Types for all the media: ``'image'`` or
                ``'text'``.
            captions (list(str), optional): Media captions to appear below
                the media.
            media_width (int, optional): Media width in pixels.
        """
        if captions is None:
            captions = [None] * len(media)
        row_str = '''
        <tr>
'''
        for x, xtype, xcap in zip(media, types, captions):
            xtype = xtype.lower()
            # Image
            if xtype == 'image':
                row_str += '''
            {td}<img src="{img}" alt="{img}" width="{media_width}">'''.format(
                td=self.td, img=x, media_width=media_width)
            # Text
            elif xtype == 'text':
                row_str += '''
            {td}<p width="{media_width}">{text}</p>'''.format(
                td=self.td, text=x, media_width=media_width)
            else:
                raise NotImplementedError(xtype)
            if xcap is not None:
                row_str += '''
            <br><p>{text}</p>'''.format(text=xcap)
            row_str += '''</td>
'''
        row_str += '''
        </tr>
'''
        self.body += row_str


if __name__ == '__main__':
    n_col = 6
    n_row = 2

    # Fake data
    medias, media_types, caps = ['Some text'], ['text'], [None]
    for col_i in range(1, n_col):
        medias.append('image%02d.png' % col_i)
        media_types.append('image')
        caps.append('Caption %d' % col_i)

    html = HTML('/usr/local/google/home/xiuming/Desktop/test.html')
    html.add_header("Hello, world!")

    # Table
    img_table = html.add_table(header=["Column %d" % x for x in range(n_col)])
    for row_i in range(n_row):
        img_table.add_row(medias, media_types, caps)

    html.save()
