# distutils: language=c++

from chimerabot.core.time_iterator cimport TimeIterator


cdef class PyTimeIterator(TimeIterator):
    pass
