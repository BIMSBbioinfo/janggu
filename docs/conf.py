# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys
#import mock

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    #'matplotlib.sphinxext.plot_directive',
    #'matplotlib.sphinxext.only_directives',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

#MOCK_MODULES = [
#'tensorflow',
#'pysam',
#'pyBigWig'
#'matplotlib',
#'seaborn',
#'dash_html_components',
#'dash_core_components',
#'dash',
#'dash_renderer',
#'scikit-learn',
#]
#
#for mod_name in MOCK_MODULES:
#    sys.modules[mod_name] = mock.Mock()

autodoc_mock_imports = ["keras", "tensorflow", "matplotlib", "sklearn", "seaborn", "Bio",
                        "pandas", "compat", "scipy", "scipy.sparse"]

if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = u'Janggu'
year = u'2017-2020'
author = u'Wolfgang Kopp'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.9.9'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/BIMSBbioinfo/janggu/issues/%s', '#'),
    'pr': ('https://github.com/BIMSBbioinfo/janggu/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
