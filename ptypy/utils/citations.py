"""
This module provides basic bibliography management.
"""

class JournalArticle(object):
    def __init__(self, author, journal, volume, year, page=None, doi=None):
        self.author = author
        self.journal = journal
        self.volume = volume
        self.year = year
        self.page = page
        self.doi = doi

    def __str__(self):
        out = ('%s, %s %s (%s)'
                % (self.author, self.journal, self.volume, str(self.year)))
        if self.page is not None:
            out += ' %s' % str(self.page)
        if self.doi is not None:
            out += ', doi: %s.' % self.doi
        else:
            out += '.'
        return out

class Bibliography(object):
    def __init__(self):
        self.entries = []

    def add_article(self, comment, *args, **kwargs):
        # avoid double entries
        for entry in self.entries:
            if entry['comment'] == comment: return None

        self.entries.append({
            'comment': comment, 
            'citation': JournalArticle(*args, **kwargs)
            })

    def __str__(self):
        out = []
        for entry in self.entries:
            out.append(entry['comment'] + ':\n    ' + str(entry['citation']))
        return '\n'.join(out)
