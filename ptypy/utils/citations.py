"""
This module provides basic bibliography management.
"""

class JournalArticle(object):
    def __init__(self, author, journal, volume, year, page, doi=None):
        self.author = author
        self.journal = journal
        self.volume = volume
        self.year = str(year)
        self.page = str(page)
        self.doi = doi

    def __str__(self):
        out = ('%s, %s %s (%s) %s'
                % (self.author, self.journal, self.volume, self.year, self.page))
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
