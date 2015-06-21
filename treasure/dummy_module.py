"""
This is a dummy module as template for modules to be placed in lib/modules

Modules are code pieces / functions that need a set (dictionary) of parameters to function
"""
from ..utils.parameters import Param
# dummy.py

# description is either a dictionary, Param instance or a file, where all defaults are explained
DESCRIPTION = Param(
structure = ('type','default','doc'),
####
sheep = ('int', 10 ,'number of sheep'),
black = ('int', 0,'those sheep are black'),
guards = ('(str,str)', ['bruno','kopernicus'], 'names (list) of dogs guarding the sheep')
)

#this one will be used as standard and updated from user templates
DEFAULT = Param(
    sheep = 10,
    black = 0,
    guards = ['bruno','kopernicus']
)
# add other
UCL = Param(
    sheep = 2,
    black = 0,
    guards = ['Pierre']
)
TUM = Param(
    guards = ['Franz','Postdocs'],
    black = 5,
    sheep = 30
)

def black_sheep_in_the_herd(pars,*args,**kwargs):
    """
    please explain every ADDITIONAL argument / keyvalue here
    """
    p = Param(DEFAULT)
    p.update(pars)
    ####
    #function body
    a = 'This group is %d members strong, has %d lazy students' % (p.sheep,p.black)
    b = '...and there for need konstan supervision by '+ ''.join(str(g)+' & ' for g in p.guards)
    return a+b
    
    

