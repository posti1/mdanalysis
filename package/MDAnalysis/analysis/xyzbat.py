# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
r"""Bond-Angle-Torsion coordinates analysis --- :mod:`MDAnalysis.analysis.bat`
===========================================================================

:Author: Soohaeng Yoo Willow and David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: 2.0.0

This module contains classes for interconverting between Cartesian and an
internal coordinate system, Bond-Angle-Torsion (BAT) coordinates
:footcite:p:`Chang2003`, for a given set of atoms or residues. This coordinate
system is designed to be complete, non-redundant, and minimize correlations
between degrees of freedom. Complete and non-redundant means that for N atoms
there will be 3N Cartesian coordinates and 3N BAT coordinates. Correlations are
minimized by using improper torsions, as described in :footcite:p:`Hikiri2016`.

More specifically, bond refers to the bond length, or distance between
a pair of bonded atoms. Angle refers to the bond angle, the angle between
a pair of bonds to a central atom. Torsion refers to the torsion angle.
For a set of four atoms a, b, c, and d, a torsion requires bonds between
a and b, b and c, and c and d. The torsion is the angle between a plane
containing atoms a, b, and c and another plane containing b, c, and d.
For a set of torsions that share atoms b and c, one torsion is defined as
the primary torsion. The others are defined as improper torsions, differences
between the raw torsion angle and the primary torsion. This definition reduces
the correlation between the torsion angles.

Each molecule also has six external coordinates that define its translation and
rotation in space. The three Cartesian coordinates of the first atom are the
molecule's translational degrees of freedom. Rotational degrees of freedom are
specified by the axis-angle convention. The rotation axis is a normalized vector
pointing from the first to second atom. It is described by the polar angle,
:math:`\phi`, and azimuthal angle, :math:`\theta`. :math:`\omega` is a third angle
that describes the rotation of the third atom about the axis.

This module was adapted from AlGDock :footcite:p:`Minh2020`.


See Also
--------
:class:`~MDAnalysis.analysis.dihedrals.Dihedral`
   class to calculate dihedral angles for a given set of atoms or residues
:func:`MDAnalysis.lib.distances.calc_dihedrals()`
   function to calculate dihedral angles from atom positions


Example applications
--------------------

The :class:`~MDAnalysis.analysis.bat.BAT` class defines bond-angle-torsion
coordinates based on the topology of an atom group and interconverts between
Cartesian and BAT coordinate systems.

For example, we can determine internal coordinates for residues 5-10
of adenylate kinase (AdK). The trajectory is included within the test data files::

   import MDAnalysis as mda
   from MDAnalysisTests.datafiles import PSF, DCD
   import numpy as np

   u = mda.Universe(PSF, DCD)

   # selection of atomgroups
   selected_residues = u.select_atoms("resid 5-10")

   from MDAnalysis.analysis.bat import BAT
   R = BAT(selected_residues)

   # Calculate BAT coordinates for a trajectory
   R.run()

After :meth:`R.run()<BAT.run>`, the coordinates can be accessed with
:attr:`R.results.bat<BAT.bat>`. The following code snippets assume that the
previous snippet has been executed.

Reconstruct Cartesian coordinates for the first frame::

   # Reconstruct Cartesian coordinates from BAT coordinates
   # of the first frame
   XYZ = R.Cartesian(R.results.bat[0,:])

   # The original and reconstructed Cartesian coordinates should all be close
   print(np.allclose(XYZ, selected_residues.positions, atol=1e-6))

Change a single torsion angle by :math:`\pi`::

   bat = R.results.bat[0,:]
   bat[bat.shape[0]-12] += np.pi
   XYZ = R.Cartesian(bat)

   # A good number of Cartesian coordinates should have been modified
   np.sum((XYZ - selected_residues.positions)>1E-5)

Store data to the disk and load it again::

   # BAT coordinates can be saved to disk in the numpy binary format
   R.save('test.npy')

   # The BAT coordinates in a new BAT instance can be loaded from disk
   # instead of using the run() method.
   Rnew = BAT(selected_residues, filename='test.npy')

   # The BAT coordinates before and after disk I/O should be close
   print(np.allclose(Rnew.results.bat, R.results.bat))


Analysis classes
----------------
.. autoclass:: BAT
    :members:
    :inherited-members:

    .. attribute:: results.bat

        Contains the time series of the Bond-Angle-Torsion coordinates as a
        (nframes, 3N) :class:`numpy.ndarray` array. Each row corresponds to
        a frame in the trajectory. In each column, the first six elements
        describe external degrees of freedom. The first three are the center
        of mass of the initial atom. The next three specify the  external angles
        according to the axis-angle convention: :math:`\phi`, the polar angle,
        :math:`\theta`, the azimuthal angle, and :math:`\omega`, a third angle
        that describes the rotation of the third atom about the axis. The next
        three degrees of freedom are internal degrees of freedom for the root
        atoms: :math:`r_{01}`, the distance between atoms 0 and 1,
        :math:`r_{12}`, the distance between atoms 1 and 2,
        and :math:`a_{012}`, the angle between the three atoms.
        The rest of the array consists of all the other bond distances,
        all the other bond angles, and then all the other torsion angles.


References
----------

.. footbibliography::

"""
import logging
import warnings

import numpy as np
import copy

import MDAnalysis as mda
from .base import AnalysisBase

from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from MDAnalysis.lib.mdamath import make_whole

from ..due import due, Doi

logger = logging.getLogger(__name__)


def _sort_atoms_by_mass(atoms, reverse=False):
    r"""Sorts a list of atoms by name and then by index

    The atom index is used as a tiebreaker so that the ordering is reproducible.

    Parameters
    ----------
    ag_o : list of Atoms
        List to sort
    reverse : bool
        Atoms will be in descending order

    Returns
    -------
    ag_n : list of Atoms
        Sorted list
    """
    return sorted(atoms, key=lambda a: (a.mass, a.index), reverse=reverse)


def _find_torsions(root, atoms):
    """Constructs a list of torsion angles

    Parameters
    ----------
    root : AtomGroup
        First three atoms in the coordinate system
    atoms : AtomGroup
        Atoms that are allowed to be part of the torsion angle

    Returns
    -------
    torsions : list of AtomGroup
        list of AtomGroup objects that define torsion angles
    """
    torsions = []
    selected_atoms = list(root)


    while len(selected_atoms) < len(atoms):
        torsionAdded = False
        for a1 in selected_atoms:
            # Find a0, which is a new atom connected to the selected atom
            a0_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                if (a in atoms) and (a not in selected_atoms))
            for a0 in a0_list:
                # Find a2, which is connected to a1, is not a terminal atom,
                # and has been selected
                a2_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                    if (a!=a0) and len(a.bonded_atoms)>1 and \
                        (a in atoms) and (a in selected_atoms))
                for a2 in a2_list:
                    # Find a3, which is
                    # connected to a2, has been selected, and is not a1
                    a3_list = _sort_atoms_by_mass(a for a in a2.bonded_atoms \
                        if (a!=a1) and \
                            (a in atoms) and (a in selected_atoms))
                    for a3 in a3_list:
                        # Add the torsion to the list of torsions
                        torsions.append(mda.AtomGroup([a0, a1, a2, a3]))
                        # Add the new atom to selected_atoms
                        # which extends the loop
                        selected_atoms.append(a0)
                        torsionAdded = True
                        break  # out of the a3 loop
                    break  # out of the a2 loop
        if torsionAdded is False:
            print('Selected atoms:')
            print([a.index + 1 for a in selected_atoms])
            print('Torsions found:')
            print([list(t.indices + 1) for t in torsions])
            raise ValueError('Additional torsions not found.')
    return torsions




class Cts(AnalysisBase):

    def __init__(self, ag, filename=None, **kwargs):
        
        super(Cts, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag


    def _prepare(self):
        self.results.cts = np.zeros(
                (self.n_frames, 3*self._ag.n_atoms), dtype=np.float64)   

    def _single_frame(self):

        cts = self._ag.positions
        self.results.cts[self._frame_index, :] = cts.flatten()


    def smooth(self, window=10):

        # make_whole makes only whole for every frame, so I want to know the boxsize for averaging
        boundary = self._get_boundaries()

        itn_COS = self._cos( self.results.itn, boundary )
        itn_SIN = self._sin( self.results.itn, boundary )

        nframes = len(boundary)
        itn_COS_AVG = np.array( [ np.mean(itn_COS[max(0,i-(window-1)//2):min(i+1+window//2, nframes-1)], axis=0) for i in range(nframes) ] )
        itn_SIN_AVG = np.array( [ np.mean(itn_SIN[max(0,i-(window-1)//2):min(i+1+window//2, nframes-1)], axis=0) for i in range(nframes) ] )

        itn_AVG = self._arctan2( itn_SIN_AVG, itn_COS_AVG, boundary )
        #return itn_AVG
        self.results.itn = itn_AVG


    def _cos(self, itn, boundary):
        cos = np.where( np.isinf(boundary), itn,  np.cos(2.0*np.pi*itn/boundary) )
        return cos
    def _sin(self, itn, boundary):
        sin = np.where( np.isinf(boundary), itn,  np.sin(2.0*np.pi*itn/boundary) )
        return sin
    def _arctan2(self, sin, cos, boundary):
        ret = np.where( np.isinf(boundary), sin, boundary*np.arctan2(sin, cos)/2.0/np.pi )
        return ret

    def _get_boundaries(self):

        traj = self._ag.universe.trajectory
        nframes = len(traj)
        natoms_TOT = len(self._ag)

        box = np.zeros((nframes, 3))
        for f,ts in enumerate(traj):
            box[f] = ts.dimensions[:3]

        return np.tile(box, natoms_TOT)


class CtsItnTrafo(AnalysisBase):
    """
    Calculate BAT coordinates for the specified AtomGroup.

    Bond-Angle-Torsions (BAT) internal coordinates will be computed for
    the group of atoms and all frame in the trajectory belonging to `ag`.

    Attributes:
        a - 
    Methods:
        b - 

    """
    
    @due.dcite(Doi("10.1002/jcc.26036"),
               description="Bond-Angle-Torsions Coordinate Transformation",
               path="MDAnalysis.analysis.bat.BAT")
    def __init__(self, ag, filename=None, **kwargs):
        """
        init transformer

        Parameters:

        ag : AtomGroup or Universe
            Group of atoms for which the BAT coordinates are calculated.
            `ag` must have a bonds attribute.
            If unavailable, bonds may be guessed using
            :meth:`AtomGroup.guess_bonds <MDAnalysis.core.groups.AtomGroup.guess_bonds>`.
            `ag` must only include one molecule.
            If a trajectory is associated with the atoms, then the computation
            iterates over the trajectory.
        filename : str
            Name of a numpy binary file containing a saved bat array.
            If filename is not ``None``, the data will be loaded from this file
            instead of being recalculated using the run() method.
        """

        super(CtsItnTrafo, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag
        self._residues = self._ag.residues

        self._known_residues = [] # this is to make it much quicker
        self._root_idx = []
        self._torsion_idx = []
        self._primary_torsion_indices = []
        self._unique_primary_torsion_indices = []
        self._ag1_idx = []
        self._ag2_idx = []
        self._ag3_idx = []
        self._ag4_idx = []


        # test if I can make this faster with list comprehension TODO
        for res in self._residues:

            idx = [ i for i, xres in enumerate(self._known_residues) if xres['resname'] == res.resname ]
            if len(idx) == 0:

                atoms = res.atoms
                natoms = len(atoms)

                if natoms > 3 : # do the original code

                    terminal_atoms = _sort_atoms_by_mass([a for a in atoms if len(a.bonds)==1], reverse=True)
                    initial_atom = terminal_atoms[0]
 
                    second_atom = initial_atom.bonded_atoms[0]

                    third_atom = _sort_atoms_by_mass(\
                        [a for a in second_atom.bonded_atoms \
                        if (a in atoms) and (a!=initial_atom) \
                        and (a not in terminal_atoms)], \
                        reverse=True)[0]

                    # I have no clue how this works but it does so I keep it even though it seems very cumbersome
                    root = mda.AtomGroup([initial_atom, second_atom, third_atom])
                    self._root_idx.append( [(atoms.indices==a.index).nonzero()[0][0] for a in root] )

                    torsions = _find_torsions(root, atoms)
                    prior_atoms = [sorted([a1, a2]) for (a0, a1, a2, a3) in torsions]
                    primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) for n in range(len(prior_atoms))]
                    self._primary_torsion_indices.append( primary_torsion_indices )
                    self._unique_primary_torsion_indices.append( list(set(primary_torsion_indices)) )

                    torsion_idx = [[(atoms.indices==a.index).nonzero()[0][0] for a in t] for t in torsions]
                    self._torsion_idx.append( torsion_idx )

                    self._ag1_idx.append( [ idx[0] for idx in torsion_idx ] )
                    self._ag2_idx.append( [ idx[1] for idx in torsion_idx ] )
                    self._ag3_idx.append( [ idx[2] for idx in torsion_idx ] )
                    self._ag4_idx.append( [ idx[3] for idx in torsion_idx ] )


                elif natoms == 3:

                    terminal_atoms = _sort_atoms_by_mass([a for a in atoms if len(a.bonds)==1], reverse=True)
                    initial_atom = terminal_atoms[0]
                    second_atom = initial_atom.bonded_atoms[0]

                    third_atom = [ a for a in second_atom.bonded_atoms if (a!=initial_atom) ][0]
                    root = mda.AtomGroup([initial_atom, second_atom, third_atom])
                    root_idx = [(atoms.indices==a.index).nonzero()[0][0] for a in root]
                    self._root_idx.append(root_idx)

                    self._torsion_idx.append( None )
                    self._primary_torsion_indices.append( None )
                    self._unique_primary_torsion_indices.append( None )

                    self._ag1_idx.append( [root_idx[0], root_idx[1]] )
                    self._ag2_idx.append( [root_idx[1], root_idx[2]] )
                    self._ag3_idx.append( [root_idx[2], root_idx[0]] )
                    self._ag4_idx.append( None )


                elif natoms == 2 :

                    terminal_atoms = _sort_atoms_by_mass(atoms, reverse=True)
                    initial_atom = terminal_atoms[0]
                    second_atom = terminal_atoms[1]
                    
                    third_atom = None
                    
                    root = mda.AtomGroup([initial_atom, second_atom])
                    root_idx = [(atoms.indices==a.index).nonzero()[0][0] for a in root]
                    self._root_idx.append(root_idx)

                    self._torsion_idx.append( None )
                    self._primary_torsion_indices.append( None )
                    self._unique_primary_torsion_indices.append( None )

                    self._ag1_idx.append( [root_idx[0]] )
                    self._ag2_idx.append( [root_idx[1]] )
                    self._ag3_idx.append( None )
                    self._ag4_idx.append( None )

                else:

                    self._root_idx.append( [0] )

                    self._torsion_idx.append( None )
                    self._primary_torsion_indices.append( None )
                    self._unique_primary_torsion_indices.append( None )

                    self._ag1_idx.append( [0] )
                    self._ag2_idx.append( None )
                    self._ag3_idx.append( None )
                    self._ag4_idx.append( None )
                
                self._known_residues.append( {'resname': res.resname,
                                            'root_idx': self._root_idx[-1],
                                            'torsion_idx': self._torsion_idx[-1],
                                            'primary_torsion_indices': self._primary_torsion_indices[-1],
                                            'unique_primary_torsion_indices': self._unique_primary_torsion_indices[-1],
                                            'ag1_idx': self._ag1_idx[-1],
                                            'ag2_idx': self._ag2_idx[-1],
                                            'ag3_idx': self._ag3_idx[-1],
                                            'ag4_idx': self._ag4_idx[-1] })

            else:

                self._root_idx.append( self._known_residues[idx[0]]['root_idx'] )
                self._torsion_idx.append( self._known_residues[idx[0]]['torsion_idx'] )
                self._primary_torsion_indices.append( self._known_residues[idx[0]]['primary_torsion_indices'] )
                self._unique_primary_torsion_indices.append( self._known_residues[idx[0]]['unique_primary_torsion_indices'] )

                self._ag1_idx.append( self._known_residues[idx[0]]['ag1_idx'] )
                self._ag2_idx.append( self._known_residues[idx[0]]['ag2_idx'] )
                self._ag3_idx.append( self._known_residues[idx[0]]['ag3_idx'] )
                self._ag4_idx.append( self._known_residues[idx[0]]['ag4_idx'] )

            if filename is not None:
                raise NotImplementedError
                #self.load(filename)


    def _prepare(self):
        self.results.itn = np.zeros(
                (self.n_frames, 3*self._ag.n_atoms), dtype=np.float64)       

    def _single_frame(self):
        # Calculate coordinates based on the root atoms
        # The rotation axis is a normalized vector pointing from atom 0 to 1
        # It is described in two degrees of freedom
        # by the polar angle and azimuth

        if self._ag.dimensions is None:
            p012 = [ res.atoms[idx].positions for res, idx in zip(self._residues, self._root_idx)  ]
        else:
            p012 = [ make_whole(res.atoms[idx], inplace=False) for res, idx in zip(self._residues, self._root_idx)  ]


        v01 = [ p[1]-p[0] if len(p)>1 else None for p in p012 ] 
        v21 = [ p[1]-p[2] if len(p)>2 else None for p in p012 ]
        # Internal coordinates
        r01 = [ np.sqrt(np.einsum('i,i->',v,v)) if v is not None else None for v in v01 ]
        # Distance between first two root atoms
        r12 = [ np.sqrt(np.einsum('i,i->',v,v)) if v is not None else None for v in v21 ]
        # Distance between second two root atoms
        # Angle between root atoms
        a012 = [ np.arccos(max(-1.,min(1.,np.einsum('i,i->',v,w)/ x/ y ))) if w is not None else None for v,w,x,y in zip(v01,v21, r01, r12) ]

        e = [ v / r if v is not None else None for v,r in zip(v01,r01)] 
        phi = [ np.arctan2(_e[1], _e[0]) if _e is not None else None for _e in e ]  # Polar angle
        theta = [ np.arccos(_e[2]) if _e is not None else None for _e in e ]  # Azimuthal angle

        # Rotation to the z axis
        cp = [ np.cos(a) if a is not None else None for a in phi ]
        sp = [ np.sin(a) if a is not None else None for a in phi ]
        ct = [ np.cos(a) if a is not None else None for a in theta ]
        st = [ np.sin(a) if a is not None else None for a in theta ]
        Rz = [ np.array([[a * c, c * b, -d], [-b, a, 0],
                       [a * d, b * d, c]]) if a is not None else None for a,b,c,d in zip(cp, sp, ct, st) ]
        pos2 = [ r.dot(p[2] -p[1]) if len(p)>2 else None for r,p in zip(Rz,p012) ]
        # Angle about the rotation axis
        omega = [ np.arctan2(p[1], p[0]) if p is not None else None for p in pos2 ]

        root_based = [ np.concatenate((p[0], [a, b, c, x, y, d])) for  p,a,b,c,x,y,d  in zip(p012,phi,theta,omega,r01,r12,a012) ]
        internals = [ self._calc_internals(i) for i,res in enumerate(self._residues) ]

        bat = np.concatenate([  np.concatenate((x[x != np.array(None)],y[y != np.array(None)])) for x,y in zip(root_based, internals) ])

        self.results.itn[self._frame_index, :] = bat



    def _calc_internals(self, idx):

        ag1_idx = self._ag1_idx[idx]
        ag2_idx = self._ag2_idx[idx]
        ag3_idx = self._ag3_idx[idx]
        ag4_idx = self._ag4_idx[idx]
        atoms = self._residues[idx].atoms

        if ag4_idx is not None :
            # Calculate internal coordinates from the torsion list
            ag1 = atoms[ag1_idx]
            ag2 = atoms[ag2_idx]
            ag3 = atoms[ag3_idx]
            ag4 = atoms[ag4_idx]
            bonds = calc_bonds(ag1.positions, ag2.positions, box=ag1.dimensions)
            angles = calc_angles(ag1.positions, ag2.positions, ag3.positions, box=ag1.dimensions)
            torsions = calc_dihedrals(ag1.positions, ag2.positions, ag3.positions, ag4.positions, box=ag1.dimensions)

            # When appropriate, calculate improper torsions
            shift = torsions[self._primary_torsion_indices[idx]]
            shift[self._unique_primary_torsion_indices[idx]] = 0.

            torsions -= shift
            # Wrap torsions to between -np.pi and np.pi
            torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

            return np.concatenate( (bonds, angles, torsions))

        else:

            return np.array([None])



    def load(self, filename, start=None, stop=None, step=None):
        """Loads the bat trajectory from a file in numpy binary format

        Parameters
        ----------
        filename : str
            name of numpy binary file
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame

        See Also
        --------
        save: Saves the bat trajectory in a file in numpy binary format
        """
        raise NotImplementedError

        logger.info("Choosing frames")
        self._setup_frames(self._trajectory, start, stop, step)

        logger.info("Loading file")
        self.results.bat = np.load(filename)

        # Check array dimensions
        if self.results.bat.shape != (self.n_frames, 3*self._ag.n_atoms):
            errmsg = ('Dimensions of array in loaded file, '
                      f'({self.results.bat.shape[0]},'
                      f'{self.results.bat.shape[1]}), differ from required '
                      f'dimensions of ({self.n_frames, 3*self._ag.n_atoms})')
            raise ValueError(errmsg)
        # Check position of initial atom
        if (self.results.bat[0, :3] != self._root[0].position).any():
            raise ValueError('Position of initial atom in file ' + \
                'inconsistent with current trajectory in starting frame.')
        return self

    def save(self, filename):

        raise NotImplementedError

        """Saves the bat trajectory in a file in numpy binary format

        See Also
        --------
        load: Loads the bat trajectory from a file in numpy binary format
        """
        np.save(filename, self.results.bat)



    def smooth(self, window=10):

        # make_whole makes only whole for every frame, so I want to know the boxsize for averaging
        boundary = self._get_boundaries()

        itn_COS = self._cos( self.results.itn, boundary )
        itn_SIN = self._sin( self.results.itn, boundary )

        nframes = len(boundary)
        itn_COS_AVG = np.array( [ np.mean(itn_COS[max(0,i-(window-1)//2):min(i+1+window//2, nframes-1)], axis=0) for i in range(nframes) ] )
        itn_SIN_AVG = np.array( [ np.mean(itn_SIN[max(0,i-(window-1)//2):min(i+1+window//2, nframes-1)], axis=0) for i in range(nframes) ] )

        itn_AVG = self._arctan2( itn_SIN_AVG, itn_COS_AVG, boundary )
        #return itn_AVG
        self.results.itn = itn_AVG


    def _cos(self, itn, boundary):
        cos = np.where( np.isinf(boundary), itn,  np.cos(2.0*np.pi*itn/boundary) )
        return cos
    def _sin(self, itn, boundary):
        sin = np.where( np.isinf(boundary), itn,  np.sin(2.0*np.pi*itn/boundary) )
        return sin
    def _arctan2(self, sin, cos, boundary):
        ret = np.where( np.isinf(boundary), sin, boundary*np.arctan2(sin, cos)/2.0/np.pi )
        return ret

    def _get_boundaries(self):

        traj = self._ag.universe.trajectory
        nframes = len(traj)
        natoms_TOT = len(self._ag)

        box = np.zeros((nframes, 3))
        for f,ts in enumerate(traj):
            box[f] =  ts.dimensions[:3]

        boundary = np.zeros((nframes, 3*natoms_TOT))
        natoms = [ res.atoms.n_atoms for res in self._residues ]
        residx = ([0] + list(3*np.cumsum(natoms)))

        for res, idx, n in zip(self._residues, residx[:-1], natoms):

            boundary[:,[idx, idx+1, idx+2]] = box

            if n == 1:
                pass
            elif n == 2:
                boundary[:,[idx+3,idx+4]] = 2.0*np.pi
                boundary[:,idx+5] = np.inf
            else:
                boundary[:,[idx+3,idx+4,idx+5,idx+8]] = 2.0*np.pi
                boundary[:,[idx+6, idx+7]] = np.inf

                if n >> 3:
                    
                    boundary[:,idx+9:idx+n+6] = np.inf # bonds
                    boundary[:,idx+n+6:idx+3*n] = 2.0*np.pi # angles

        return boundary


    def reverse(self):
        """Conversion of a single frame from BAT to Cartesian coordinates

        One application of this function is to determine the new
        Cartesian coordinates after modifying a specific torsion angle.

        Parameters
        ----------
        bat : numpy.ndarray
            an array with dimensions (3N,) with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.

        Returns
        -------
        XYZ : numpy.ndarray
            an array with dimensions (N,3) with Cartesian coordinates. The first
            dimension has the same ordering as the AtomGroup used to initialize
            the class. The molecule will be whole opposed to wrapped around a
            periodic boundary.
        """
        # Split the bat vector into more convenient variables

        # first split everything up again so I have a list of residues each (nframes, natoms)
        bat = self.results.itn
        nframes = len(bat)

        natoms = [ res.atoms.n_atoms for res in self._residues ]
        residx = ([0] + list(3*np.cumsum(natoms)))
        internals = [ bat[:,residx[i]:residx[i+1]] for i in range(len(residx)-1) ]

        origin = [ itn[:,:3] for itn in internals ]
        phi =   [ itn[:,3] if len(itn[0])>3 else None for itn in internals]
        theta = [ itn[:,4] if len(itn[0])>3 else None for itn in internals]
        omega = [ itn[:,5] if len(itn[0])>3 else None for itn in internals]

        r01 =  [ itn[:,6] if len(itn[0])>6 else None for itn in internals]
        r12 =  [ itn[:,7] if len(itn[0])>6 else None for itn in internals]
        a012 = [ itn[:,8] if len(itn[0])>6 else None for itn in internals]

        # Set initial root atom positions based on internal coordinates
        p0 = origin
        p1 = [ np.column_stack((np.zeros(nframes), np.zeros(nframes), x)) if x is not None else None for x in r01 ]
        p2 = [ np.column_stack((y * np.sin(a), np.zeros(nframes), x - y * np.cos(a))) if x is not None else None for x,y,a in zip(r01,r12,a012) ]


        # Rotate the third atom by the appropriate value
        co = [ np.cos(a) if a is not None else None for a in omega ]
        so = [ np.sin(a) if a is not None else None for a in omega ]
        # $R_Z(\omega)$

        Romega = [ np.array([[c, -s, np.zeros(nframes)], [s, c, np.zeros(nframes)], [np.zeros(nframes), np.zeros(nframes), np.ones(nframes)]]) if c is not None else None for s,c in zip(so, co) ]

        p2 = [ np.einsum("ijk,kj->ki", r, p) if p is not None else None for r,p in zip(Romega,p2) ]


        # Rotate the second two atoms to point in the right direction
        cp = [ np.cos(a) if a is not None else None for a in phi ]
        sp = [ np.sin(a) if a is not None else None for a in phi ]
        ct = [ np.cos(a) if a is not None else None for a in theta ]
        st = [ np.sin(a) if a is not None else None for a in theta ]
        # $R_Z(\phi) R_Y(\theta)
        Re = [ np.array([[a * b, -c, a * d], [b * c, a, c * d],
                       [-d, np.zeros(nframes), b]]) if a is not None else None for a,b,c,d in zip(cp, ct, sp, st) ]


        p1 = [ np.einsum("ijk,kj->ki", r, p) + o if p is not None else None for r,p,o in zip(Re,p1,origin) ]   
        p2 = [ np.einsum("ijk,kj->ki", r, p) + o if p is not None else None for r,p,o in zip(Re,p2,origin) ]         


        XYZ = [ self._place_remaining_atoms(x, y, z, itn, ri, ti, pti, upti) for x, y, z, itn, ri, ti, pti, upti in zip(p0, p1, p2, internals, self._root_idx, self._torsion_idx, self._primary_torsion_indices, self._unique_primary_torsion_indices) ]

        coords = np.concatenate(XYZ, axis=1)

        return coords


    def _place_remaining_atoms(self, p0, p1, p2, internal_coords, root_idx, torsion_idx, 
                               primary_torsion_indices, unique_primary_torsion_indices):

        natoms = len(internal_coords[0])//3
        nframes = len(internal_coords)

        if natoms == 1 :

            return p0


        elif natoms == 2:
 
            r = internal_coords[:,5]
            phi = internal_coords[:,3]
            theta = internal_coords[:,4]
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            XYZ = np.zeros((nframes, 2, 3))
            XYZ[:,root_idx[0]] = p0
            XYZ[:,root_idx[1]] = p0 + np.column_stack((x,y,z))

            return XYZ.reshape((nframes, 6))


        elif natoms == 3:

            XYZ = np.zeros((nframes, 3, 3))
            XYZ[:,root_idx[0]] = p0
            XYZ[:,root_idx[1]] = p1
            XYZ[:,root_idx[2]] = p2

            return  XYZ.reshape((nframes, 9))


        else:

            bonds = internal_coords[:,9:natoms+6].T
            angles = internal_coords[:,natoms+6:2*natoms+3].T
            torsions = copy.deepcopy(internal_coords[:,2*natoms+3:]).T

            # When appropriate, convert improper to proper torsions
            shift = torsions[primary_torsion_indices]
            shift[unique_primary_torsion_indices] = 0.
            torsions += shift
            torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi


            XYZ = np.zeros((nframes, natoms,3))
            XYZ[:,root_idx[0]] = p0
            XYZ[:,root_idx[1]] = p1
            XYZ[:,root_idx[2]] = p2

            for ((a0,a1,a2,a3), r01, angle, torsion) \
                in zip(torsion_idx, bonds, angles, torsions):

                p1 = XYZ[:,a1]
                p3 = XYZ[:,a3]
                p2 = XYZ[:,a2]

                sn_ang = np.sin(angle)
                cs_ang = np.cos(angle)
                sn_tor = np.sin(torsion)
                cs_tor = np.cos(torsion)

                v21 = p1 - p2
                v21 = np.divide(v21.T, np.sqrt(np.einsum('ij,ij->i',v21,v21))).T
                v32 = p2 - p3
                v32 = np.divide( v32.T, np.sqrt(np.einsum('ij,ij->i',v32,v32))).T

                vp = np.cross(v32, v21)
                cs = np.einsum('ij,ij->i',v21,v32)

                sn = np.maximum(np.sqrt(1.0 - cs * cs), 0.0000000001)
                vp = np.divide( vp.T,  sn ).T
                vu = np.cross(vp, v21)

                XYZ[:,a0] = p1 + (r01*( vu.T*sn_ang.T*cs_tor + vp.T*sn_ang*sn_tor - v21.T*cs_ang )).T

            return XYZ.reshape((nframes,3*natoms))
            


    @property
    def atoms(self):
        """The atomgroup for which BAT are computed (read-only property)"""
        return self._ag
