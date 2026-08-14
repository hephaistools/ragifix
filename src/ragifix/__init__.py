"""ragifix — service RAG auto-porté.

Un seul process qui expose, via une API HTTP locale, l'ajout, la
suppression et l'interrogation de documents. Aucune notion de "source" :
c'est le rôle de ragifix-collector (ou de tout autre client de cette API).
"""

__version__ = "0.1.0"
