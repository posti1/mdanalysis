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
r"""TITLE --- :mod:`MDAnalysis.analysis.xyzbat`
===========================================================================

:Author: Soohaeng Yoo Willow and David Minh AND ME?
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: X.X.X

XXXX


AlGDock :footcite:p:`Minh2020`.


See Also
--------
XXXX


Example applications
--------------------
XXXX


Analysis classes
----------------
XXX


References
----------
XXX

"""


import logging
logger = logging.getLogger(__name__)

import numpy as np
import copy

import MDAnalysis as mda
from .base import AnalysisBase

from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.exceptions import MissingDataWarning#NoDataError#SelectionError

from ..due import due, Doi



class CartesianAndInternalRepresentation(AnalysisBase):
    """
    Class for switching between Cartesian and internal coordinates

    XXXs

    Attributes:
        atoms - AtomGroup
                All atoms used ion the analysis (read only)
        n_frames - int
                number of frames
    Methods:
        b -

    """

    @due.dcite(Doi("10.1002/jcc.26036"),
               description="Bond-Angle-Torsions Coordinate Transformation",
               path="MDAnalysis.analysis.bat.BAT")
    def __init__(self, ag, filename=None, group="residues", **kwargs):
        """
        init transformer

        Parameters:
          ag        - AtomGroup or Universe  TODO: Works for Universe ???
                      XXX
          filename  - str
                      TODO!!
          group     - str [default=`residues`]
                      Mode of grouping atoms together. Choose between `residues`
                      (each residue individually) or `molecules`/`fragments`
                      (each molecule or fragment).
        """

        super(CartesianAndInternalRepresentation, self).__init__(ag.universe.trajectory, **kwargs)
        self._all_atoms = ag

        # The previous BAT calculator could only be applied to one molecule.
        # Here, I changed it in a way that you can read multiple molecules
        # of the same species without always drawing the same tree again.
        self._known_residues = []
        self._root_idx = []
        self._torsion_idx = []
        self._has_run_once = False

        # If we load a trajectory without any residue / fragment information
        # (e.g. with a GRO file), then we have no information about
        # masses, bonds, etc., which makes most of this here pointless
        if not hasattr(self._all_atoms, 'bonds'):
            raise MissingDataWarning('Universe contains no information ' \
                                     'about bonds, angles and torsions')
            self._contains_only_xyz_info = True
            self._atom_groups = [ self._all_atoms ]
        else:
            self._contains_only_xyz_info = False
            if group == 'residues':
                # we may have only part of a residue, which we worry
                # about later
                self._atom_groups = [ mda.AtomGroup([ atom \
                                            for atom in res.atoms \
                                            if atom in self._all_atoms  ]) 
                                        for res in self._all_atoms.residues ]
            elif group == 'fragments':
                self._atom_groups = [ mda.AtomGroup([ atom \
                                            for atom in frag.atoms \
                                            if atom in self._all_atoms ])
                                        for frag in self._all_atoms.fragments ]
            else:
                # TODO what is the difference of fragments and molecules?
                # is there another corner case?
                raise NotImplementedError


        # test if I can make this faster with list comprehension TODO
        if self._contains_only_xyz_info:
            self._atom_groups_trees = None
        else:
            self._atom_groups_trees = [ self._init_tree(res) for res in self._atom_groups]

        if filename is not None:
            raise NotImplementedError


    def _init_tree(self, res):

        # first check if we already initialized a residue of that type
        # then, we only have to copy the tree and do not need to calculate it
        idx = [ i for i, r in enumerate(self._known_residues) \
            if _are_same_lists(r['resnames'],res.resnames) and _are_same_lists(r['names'],res.names) ]

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
                    root = mda.AtomGroup([initial_atom, second_atom, third_atom])
                    self._root_idx.append( [(atoms.indices==a.index).nonzero()[0][0] for a in root] )

                    torsions = _find_torsions(root, atoms)
                    # TODO: check if below is faster. Maybe add some check like if dihedrals exists
                    # all_torsions = [ mda.AtomGroup([a0, a1, a2, a3]) for a0,a1,a2,a3 \
                    #    in zip(atoms.dihedrals.atom1, atoms.dihedrals.atom2, atoms.dihedrals.atom3, atoms.dihedrals.atom4) ]
                    # torsions_TEST = [ t for t in all_torsions if not any(a for a in t if a not in atoms ) ]

                    torsion_idx = [[(atoms.indices==a.index).nonzero()[0][0] for a in t] for t in torsions]

                    self._torsion_idx.append( torsion_idx )

            elif natoms == 3:
                    terminal_atoms = _sort_atoms_by_mass([a for a in atoms if len(a.bonds)==1], reverse=True)
                    initial_atom = terminal_atoms[0]
                    second_atom = initial_atom.bonded_atoms[0]
                    third_atom = [ a for a in second_atom.bonded_atoms if (a!=initial_atom) ][0]
                    root = mda.AtomGroup([initial_atom, second_atom, third_atom])
                    root_idx = [(atoms.indices==a.index).nonzero()[0][0] for a in root]
                    self._root_idx.append(root_idx)
                    self._torsion_idx.append( None )

            elif natoms == 2 :
                    terminal_atoms = _sort_atoms_by_mass(atoms, reverse=True)
                    initial_atom = terminal_atoms[0]
                    second_atom = terminal_atoms[1]
                    third_atom = None
                    root = mda.AtomGroup([initial_atom, second_atom])
                    root_idx = [(atoms.indices==a.index).nonzero()[0][0] for a in root]
                    self._root_idx.append(root_idx)
                    self._torsion_idx.append( None )

            else:
                    self._root_idx.append( [0] )
                    self._torsion_idx.append( None )

            self._known_residues.append( {'resnames': res.resnames,
                                          'names':    res.names,
                                          'root_idx': self._root_idx[-1],
                                          'torsion_idx': self._torsion_idx[-1] })

        else:
            self._root_idx.append( self._known_residues[idx[0]]['root_idx'] )
            self._torsion_idx.append( self._known_residues[idx[0]]['torsion_idx'] )


    # TODO add more info stuff
    @property
    def atoms(self):
        '''The atomgroup for which BAT are computed (read-only)'''
        return self._all_atoms

    @property
    def atom_connections(self): # TODO BETTER NAME
        '''TODO'''
        return self._atom_group # we want to return the Z matrix or something


    # we inherit n_frames from the base analysis class
    def _prepare(self):
        '''inits the dict where the coords are stored'''
        self._has_run_once = True
        self._trajectory = self._all_atoms.universe.trajectory
        self.n_frames = len(self._trajectory)
        self.results.xyz = np.full((self.n_frames, 3*self._all_atoms.n_atoms), np.nan)
        self.results.itn = np.full((self.n_frames, 3*self._all_atoms.n_atoms), np.nan)


    def _single_frame(self, mode='all'):
        '''computes BATs of one frame'''

        if self._all_atoms.dimensions is None:
            p012 = [ res.atoms[idx].positions for res, idx in zip(self._atom_groups, self._root_idx)  ]
        else:
            p012 = [ make_whole(res.atoms[idx], inplace=False) for res, idx in zip(self._atom_groups, self._root_idx)  ]    

        v01 = [ p[1]-p[0] if len(p)>1 else None for p in p012 ]
        v21 = [ p[1]-p[2] if len(p)>2 else None for p in p012 ]
        # Internal coordinates
        r01 = [ np.sqrt(np.einsum('i,i->',v,v)) if v is not None else None for v in v01 ]
        # Distance between first two root atoms
        r12 = [ np.sqrt(np.einsum('i,i->',v,v)) if v is not None else None for v in v21 ]
        # Distance between second two root atoms
        # Angle between root atoms
        a012 = [ np.arccos(max(-1.,min(1.,np.einsum('i,i->',v,w)/ x/ y ))) \
                if w is not None else None for v,w,x,y in zip(v01,v21, r01, r12) ]

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
        internals = [ self._calc_BAT_using_AGs(i) for i,res in enumerate(self._atom_groups) ]

        bat = np.concatenate([  np.concatenate((x[x != np.array(None)], y[y != np.array(None)])) \
                for x,y in zip(root_based, internals) ])
        xyz = self._all_atoms.positions.flatten()

        self.results.itn[self._frame_index, :] = bat
        self.results.xyz[self._frame_index, :] = xyz


    # I overwrote the old run function.
    def run(self):
        self._prepare()
        self._run_all_at_once()

    # attempt to be faster than the run() function
    def _run_all_at_once(self):

        self.results.xyz = self._get_xyz_from_traj()
        self.results.itn = self._compute_itn_from_xyz(self.results.xyz)



    # For some reason I do not find a function which does this
    def _get_xyz_from_traj(self):

        atoms = self._all_atoms
        traj = self._trajectory
        natoms = len(atoms)

        xyz = np.empty((self.n_frames, 3*natoms), dtype=np.float64)

        for f,ts in enumerate(traj):
            xyz[f] = atoms.positions.flatten()

        return xyz


    @property
    def cartesian_coordinates(self):
        '''cartesian coordinates'''
        if self._has_run_once == False:
            self.run()
        return self.results.xyz

    @property
    def internal_coordinates(self):
        '''cartesian coordinates'''
        if self._has_run_once == False:
            self.run()
        return self.results.itn



    def load(self, filename, start=None, stop=None, step=None):
        '''
        NOT YET IMPLEMENTED!
        '''
        raise NotImplementedError

    def save(self, filename):
        '''
        NOT YET IMPLEMENTED!
        '''
        raise NotImplementedError


    def _update_coordinates_in_atom_group(self):

        traj = self._trajectory
        xyz = self.results.xyz

        natoms = [ res.atoms.n_atoms for res in self._atom_groups ]
        residx = ([0] + list(3*np.cumsum(natoms)))
        xyz_split = [ xyz[:,residx[i]:residx[i+1]] for i in range(len(residx)-1) ]

        for f,ts in enumerate(traj):

            print(f, xyz[f,0:3], self._all_atoms.positions[0] )

            self._all_atoms.positions = xyz[f].reshape((len(self._all_atoms),3))

            print(f, xyz[f,0:3], self._all_atoms.positions[0] )

            for r, n, res in zip(xyz_split, natoms, self._atom_groups):
                res.atoms.positions = r[f].reshape((n,3))

    
    def smooth_cartesian_coordinates(self, window=10):
        '''
        smooth the trajectory in terms of its Cartesian coordinates

        parameters
        ----------
        window : int
                 number of frames we average over
        '''

        xyz = self.results.xyz

        self.results.xyz = np.array( [ np.mean(xyz[max(0,i-(window-1)//2):min(i+1+window//2, self.n_frames-1)], 
                                        axis=0) for i in range(self.n_frames) ] )
        #print(self.results.xyz[:,0])
        #self._update_coordinates_in_atom_group()
        #self.run()


    def smooth_internal_coordinates(self, window=10):
        '''
        smooth the trajectory in terms of its internal coordinates

        parameters
        ----------
        window : int
                 number of frames we average over
        '''

        # make_whole makes only whole for every frame, so I want to know the boxsize for averaging
        boundary = self._get_PBC_itn()

        itn_COS = _cos( self.results.itn, boundary )
        itn_SIN = _sin( self.results.itn, boundary )

        itn_COS_AVG = np.array( [ np.mean(itn_COS[max(0,i-(window-1)//2):min(i+1+window//2, self.n_frames-1)], 
                                    axis=0) for i in range(self.n_frames) ] )
        itn_SIN_AVG = np.array( [ np.mean(itn_SIN[max(0,i-(window-1)//2):min(i+1+window//2, self.n_frames-1)], 
                                    axis=0) for i in range(self.n_frames) ] )

        self.results.itn = _arctan2( itn_SIN_AVG, itn_COS_AVG, boundary )
        # also update the cartesian coordinates
        self.results.xyz = self._compute_xyz_from_itn(self.results.itn)
        
        #self._update_coordinates()


    def _get_PBC_xyz(self):

        traj = self._trajectory
        natoms_TOT = len(self._all_atoms)

        box = np.zeros((self.n_frames, 6))
        for f,ts in enumerate(traj):
            box[f] =  ts.dimensions
        return box


    def _get_PBC_itn(self):

        traj = self._trajectory
        natoms_TOT = len(self._all_atoms)

        box = np.zeros((self.n_frames, 3))
        for f,ts in enumerate(traj):
            box[f] =  ts.dimensions[:3]

        boundary = np.zeros((self.n_frames, 3*natoms_TOT))
        natoms = [ res.atoms.n_atoms for res in self._atom_groups ]
        residx = ([0] + list(3*np.cumsum(natoms)))

        for res, idx, n in zip(self._atom_groups, residx[:-1], natoms):

            boundary[:,[idx, idx+1, idx+2]] = box

            if n == 1:
                pass
            elif n == 2:
                boundary[:,[idx+3,idx+4]] = 2.0*np.pi
                boundary[:,idx+5] = np.inf
            else:
                boundary[:,[idx+3,idx+4,idx+5,idx+8]] = 2.0*np.pi
                boundary[:,[idx+6, idx+7]] = np.inf
                if n > 3:
                    boundary[:,idx+9:idx+n+6] = np.inf # bonds
                    boundary[:,idx+n+6:idx+3*n] = 2.0*np.pi # angles

        return boundary


    def _compute_itn_from_xyz(self, xyz):
        '''
        Compute Internal from Cartesian Coordinates
        '''

        nothing = np.empty(shape=(self.n_frames,0))

        natoms = [ res.atoms.n_atoms for res in self._atom_groups ]
        residx = ([0] + list(3*np.cumsum(natoms)))

        cartesians = [ xyz[:,residx[i]:residx[i+1]] for i in range(len(residx)-1) ]

        atom_idx_abs = [ [[ 3*_r+i, 3*_r+i+1, 3*_r+i+2 ] for _r in r ] \
                for i,r in zip(residx[:-1], self._root_idx) ] # index of x atom

        p012 = [ xyz[:,idx] for idx in atom_idx_abs ]

        v01 = [ p[:,1]-p[:,0] if len(p[0])>1 else nothing for p in p012 ]
        v21 = [ p[:,1]-p[:,2] if len(p[0])>2 else nothing for p in p012 ]

        r01 = [ np.sqrt(np.einsum('ij,ij->i',v,v)) if v is not nothing else nothing for v in v01 ]
        r12 = [ np.sqrt(np.einsum('ij,ij->i',v,v)) if v is not nothing else nothing for v in v21 ]


        a012 = [ np.arccos(np.maximum(-1.,np.minimum(1.,np.einsum('ij,ij->i',v,w)/ x/ y ))) \
                    if w is not nothing else nothing for v,w,x,y in zip(v01,v21, r01, r12) ]


        e = [ (v.T / r).T if v is not nothing else nothing for v,r in zip(v01,r01)]
        phi = [ np.arctan2(_e[:,1], _e[:,0]) if _e is not nothing else nothing for _e in e ]  # Polar angle
        theta = [ np.arccos(_e[:,2]) if _e is not nothing else nothing for _e in e ]  # Azimuthal angle

        cp = [ np.cos(a) if a is not nothing else nothing for a in phi ]
        sp = [ np.sin(a) if a is not nothing else nothing for a in phi ]
        ct = [ np.cos(a) if a is not nothing else nothing for a in theta ]
        st = [ np.sin(a) if a is not nothing else nothing for a in theta ]

        Rz = [ np.stack([[a * c, c * b, -d], [-b, a, np.zeros(self.n_frames)], [a * d, b * d, c]]) \
                if a is not nothing else nothing for a,b,c,d in zip(cp, sp, ct, st) ]

        pos2 = [ np.einsum('ijk,kj->ki',r,p[:,2] -p[:,1]) if len(p[0])>2 else nothing for r,p in zip(Rz,p012) ]

        omega = [ np.arctan2(p[:,1], p[:,0]) if p is not nothing else nothing for p in pos2 ]

        root_based = [ np.column_stack((p[:,0], a, b, c, x, y, d)) for p,a,b,c,x,y,d  in zip(p012,phi,theta,omega,r01,r12,a012) ]

        box = self._get_PBC_xyz()
        internals = [ self._calc_BAT_using_xyz(i, r, box) for i,r in enumerate(cartesians) ]

        bat = np.column_stack(root_based+internals)

        return bat


    def _compute_xyz_from_itn(self, bat):
        '''
        Compute Cartesian from Internal Coordinates
        '''
        # Split the bat vector into more convenient variables

        # first split everything up again so I have a list of residues each (nframes, natoms)
        nothing = np.empty(shape=(self.n_frames,0))
        zeros = np.zeros(self.n_frames)

        natoms = [ res.atoms.n_atoms for res in self._atom_groups ]
        residx = ([0] + list(3*np.cumsum(natoms)))

        internals = [ bat[:,residx[i]:residx[i+1]] for i in range(len(residx)-1) ]

        origin = [ itn[:,:3] for itn in internals ]
        phi =   [ itn[:,3] if len(itn[0])>3 else nothing for itn in internals]
        theta = [ itn[:,4] if len(itn[0])>3 else nothing for itn in internals]
        omega = [ itn[:,5] if len(itn[0])>3 else nothing for itn in internals]

        r01 =  [ itn[:,6] if len(itn[0])>6 else nothing for itn in internals]
        r12 =  [ itn[:,7] if len(itn[0])>6 else nothing for itn in internals]
        a012 = [ itn[:,8] if len(itn[0])>6 else nothing for itn in internals]


        # Set initial root atom positions based on internal coordinates
        p0 = origin
        p1 = [ np.column_stack((zeros, zeros, x)) if x is not nothing else nothing for x in r01 ]
        p2 = [ np.column_stack((y * np.sin(a), zeros, x - y * np.cos(a))) \
                if x is not nothing else nothing for x,y,a in zip(r01,r12,a012) ]


        # Rotate the third atom by the appropriate value
        co = [ np.cos(a) if a is not nothing else nothing for a in omega ]
        so = [ np.sin(a) if a is not nothing else nothing for a in omega ]
        # $R_Z(\omega)$

        Romega = [ np.array([[c, -s, zeros], 
                             [s, c,zeros], 
                             [zeros, zeros, zeros]]) \
                        if c is not nothing else nothing for s,c in zip(so, co) ]

        p2 = [ np.einsum("ijk,kj->ki", r, p) if p is not nothing else nothing for r,p in zip(Romega,p2) ]


        # Rotate the second two atoms to point in the right direction
        cp = [ np.cos(a) if a is not nothing else nothing for a in phi ]
        sp = [ np.sin(a) if a is not nothing else nothing for a in phi ]
        ct = [ np.cos(a) if a is not nothing else nothing for a in theta ]
        st = [ np.sin(a) if a is not nothing else nothing for a in theta ]
        # $R_Z(\phi) R_Y(\theta)
        Re = [ np.array([[a * b, -c, a * d], [b * c, a, c * d],
                       [-d, zeros, b]]) if a is not nothing else nothing for a,b,c,d in zip(cp, ct, sp, st) ]


        p1 = [ np.einsum("ijk,kj->ki", r, p) + o if p is not nothing else nothing for r,p,o in zip(Re,p1,origin) ]
        p2 = [ np.einsum("ijk,kj->ki", r, p) + o if p is not nothing else nothing for r,p,o in zip(Re,p2,origin) ]

        XYZ = [ self._place_remaining_atoms(i, x, y, z, itn) for i, (x, y, z, itn) in enumerate(zip(p0, p1, p2, internals)) ]
        coords = np.concatenate(XYZ, axis=1)

        return coords



    def _calc_BAT_using_xyz(self, idx, xyz, box=None):

        if len(xyz[0]) > 9 :

            atoms = self._atom_groups[idx].atoms
            ag1 = atoms[[ c[0] for c in self._torsion_idx[idx] ]]
            ag2 = atoms[[ c[1] for c in self._torsion_idx[idx] ]]
            ag3 = atoms[[ c[2] for c in self._torsion_idx[idx] ]]
            ag4 = atoms[[ c[3] for c in self._torsion_idx[idx] ]]

            ag1_xyz = np.stack([ xyz[:,[3*c[0], 3*c[0]+1, 3*c[0]+2]] for c in self._torsion_idx[idx] ])
            ag2_xyz = np.stack([ xyz[:,[3*c[1], 3*c[1]+1, 3*c[1]+2]] for c in self._torsion_idx[idx] ])
            ag3_xyz = np.stack([ xyz[:,[3*c[2], 3*c[2]+1, 3*c[2]+2]] for c in self._torsion_idx[idx] ])
            ag4_xyz = np.stack([ xyz[:,[3*c[3], 3*c[3]+1, 3*c[3]+2]] for c in self._torsion_idx[idx] ])

            bonds = np.stack([ calc_bonds(ag1_xyz[:,t], ag2_xyz[:,t], box=box[t]) for t in range(self.n_frames) ])
            angles = np.stack([ calc_angles(ag1_xyz[:,t], ag2_xyz[:,t], ag3_xyz[:,t], box=box[t]) for t in range(self.n_frames) ])  
            torsions = np.stack([ calc_dihedrals(ag1_xyz[:,t], ag2_xyz[:,t], 
                                    ag3_xyz[:,t], ag4_xyz[:,t], box=box[t]) for t in range(self.n_frames) ])

            prior_atoms = [sorted([a1, a2]) for (a1, a2) in zip(ag2,ag3)]
            primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) for n in range(len(prior_atoms))]
            unique_primary_torsion_indices = list(set(primary_torsion_indices))

            # When appropriate, calculate improper torsions
            shift = torsions[:,primary_torsion_indices]
            shift[:,unique_primary_torsion_indices] = 0.

            torsions -= shift
            # Wrap torsions to between -np.pi and np.pi
            torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

            return np.column_stack( (bonds, angles, torsions))
        else:
            return np.empty(shape=(self.n_frames,0))


    def _calc_BAT_using_AGs(self, idx):

        if self._torsion_idx[idx] is not None :

            atoms = self._atom_groups[idx].atoms

            # Calculate internal coordinates from the torsion list
            ag1 = atoms[[ c[0] for c in self._torsion_idx[idx] ]]
            ag2 = atoms[[ c[1] for c in self._torsion_idx[idx] ]]
            ag3 = atoms[[ c[2] for c in self._torsion_idx[idx] ]]
            ag4 = atoms[[ c[3] for c in self._torsion_idx[idx] ]]
            bonds = calc_bonds(ag1.positions, ag2.positions, box=ag1.dimensions)
            angles = calc_angles(ag1.positions, ag2.positions, ag3.positions, box=ag1.dimensions)
            torsions = calc_dihedrals(ag1.positions, ag2.positions, ag3.positions, ag4.positions, box=ag1.dimensions)

            prior_atoms = [sorted([a1, a2]) for (a1, a2) in zip(ag2,ag3)]
            primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) for n in range(len(prior_atoms))]
            unique_primary_torsion_indices = list(set(primary_torsion_indices))

            # When appropriate, calculate improper torsions
            shift = torsions[primary_torsion_indices]
            shift[unique_primary_torsion_indices] = 0.

            torsions -= shift
            # Wrap torsions to between -np.pi and np.pi
            torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

            return np.concatenate( (bonds, angles, torsions))
        else:
            return np.array([None])


    def _place_remaining_atoms(self, idx, p0, p1, p2, internal_coords):

        root_idx = self._root_idx[idx]
        natoms = len(internal_coords[0])//3

        if natoms == 1 :
            return p0

        elif natoms == 2:
            r = internal_coords[:,5]
            phi = internal_coords[:,3]
            theta = internal_coords[:,4]
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            XYZ = np.zeros((self.n_frames, 2, 3))
            XYZ[:,root_idx[0]] = p0
            XYZ[:,root_idx[1]] = p0 + np.column_stack((x,y,z))

            return XYZ.reshape((self.n_frames, 6))

        elif natoms == 3:

            XYZ = np.zeros((self.n_frames, 3, 3))
            XYZ[:,root_idx[0]] = p0
            XYZ[:,root_idx[1]] = p1
            XYZ[:,root_idx[2]] = p2

            return  XYZ.reshape((self.n_frames, 9))


        else:

            torsion_idx = self._torsion_idx[idx]

            atoms = self._atom_groups[idx].atoms
            ag2 = atoms[[ c[1] for c in torsion_idx ]]
            ag3 = atoms[[ c[2] for c in torsion_idx ]]
            prior_atoms = [sorted([a1, a2]) for (a1, a2) in zip(ag2,ag3)]
            primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) for n in range(len(prior_atoms))]
            unique_primary_torsion_indices = list(set(primary_torsion_indices))
            
            bonds = internal_coords[:,9:natoms+6].T
            angles = internal_coords[:,natoms+6:2*natoms+3].T
            torsions = copy.deepcopy(internal_coords[:,2*natoms+3:]).T

            # When appropriate, convert improper to proper torsions
            shift = torsions[primary_torsion_indices]
            shift[unique_primary_torsion_indices] = 0.
            torsions += shift
            torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi


            XYZ = np.zeros((self.n_frames, natoms,3))
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
                d = np.sqrt(np.einsum('ij,ij->i',v21,v21))
                v21 = np.divide(v21.T, d, out=v21.T, where=(d!=0.0)).T
                v32 = p2 - p3

                d = np.sqrt(np.einsum('ij,ij->i',v32,v32))
                v32 = np.divide( v32.T, d, out=v32.T, where=(d!=0.0)).T

                vp = np.cross(v32, v21)
                cs = np.einsum('ij,ij->i',v21,v32)

                sn = np.maximum(np.sqrt(1.0 - cs * cs), 0.0)
                vp = np.divide( vp.T,  sn, out=vp.T, where=(sn!=0.0)).T
                vu = np.cross(vp, v21)

                XYZ[:,a0] = p1 + (r01*( vu.T*sn_ang.T*cs_tor + vp.T*sn_ang*sn_tor - v21.T*cs_ang )).T

            return XYZ.reshape((self.n_frames,3*natoms))

XyzBat = CartesianAndInternalRepresentation



# cosine of x, but refactored to have a certain period, with inf yust returning
# the argument. This is useful for calculating circular means
def _cos(x, period):
    '''returns cosine with period, returns argument if inf'''
    return np.where( np.isinf(period), x,  np.cos(2.0*np.pi*x/period) )

def _sin(x, period):
    '''returns cosine with period, returns argument if inf'''
    return np.where( np.isinf(period), x,  np.sin(2.0*np.pi*x/period) )

# arctan2 does the back-trafo from sin/cos to the original circular coordinates
# this uses the NUMPY convention, not that of wikipedia
def _arctan2(y, x, period):
    '''returns arctan or the first argument if inf'''
    return np.where( np.isinf(boundary), y, boundary*np.arctan2(y, x)/2.0/np.pi )


def _sort_atoms_by_mass(atoms, reverse=False):
    '''Sorts a list of atoms by name, then by index'''
    return sorted(atoms, key=lambda a: (a.mass, a.index), reverse=reverse)


def _are_same_lists(A,B):
    if len(A) != len(B):
        return False
    else:
        return (A==B).all()

# This is where the magic happens ...
# We want to draw a tree with our atoms which looks like this:
def _find_torsions(root, atoms):
    '''
    Constructs a list of torsion angles

    EXPLAIN HOW THIS WORKS !!!
    '''
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
