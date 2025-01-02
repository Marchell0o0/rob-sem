#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(
                rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        theta = np.linalg.norm(v)
        axis = v / theta
        skew_axis = SO3.skew(axis)
        t.rot = np.eye(3) + np.sin(theta) * skew_axis + \
            (1 - np.cos(theta)) * (skew_axis @ skew_axis)
        return t

    @staticmethod
    def skew(v: ArrayLike) -> np.ndarray:
        """Convert a 3D vector to a skew-symmetric matrix."""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        v = np.zeros(3)
        if np.allclose(self.rot, np.eye(3)):
            return v
        elif np.trace(self.rot) == -1:
            theta = np.pi
            if self.rot[2, 2] != -1:
                w = 1/(np.sqrt(2*(1 + self.rot[2, 2]))) * np.array(
                    [self.rot[0, 2], self.rot[1, 2], 1 + self.rot[2, 2]])
            elif self.rot[1, 1] != -1:
                w = 1/(np.sqrt(2*(1 + self.rot[1, 1]))) * np.array(
                    [self.rot[0, 1], 1 + self.rot[1, 1], self.rot[2, 1]])
            elif self.rot[0, 0] != -1:
                w = 1/(np.sqrt(2*(1 + self.rot[0, 0]))) * np.array(
                    [1 + self.rot[0, 0], self.rot[1, 0], self.rot[2, 0]])
            else:
                w = np.array([0, 0, 0])
            return w * theta
        else:
            theta = np.arccos((np.trace(self.rot) - 1) / 2)
            w_skew = 1/(2 * np.sin(theta)) * (self.rot - self.rot.T)
            w = SO3.unskew(w_skew)
        v = w * theta
        return v

    @staticmethod
    def unskew(w_skew: np.ndarray) -> np.ndarray:
        """Convert a skew-symmetric matrix to a 3D vector."""
        return np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        result = SO3()
        result.rot = self.rot @ other.rot
        return result

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        result = SO3()
        result.rot = self.rot.T
        return result

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        result = SO3()
        result.rot = np.array([[1, 0, 0], [0, np.cos(
            angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        return result

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        result = SO3()
        result.rot = np.array([[np.cos(angle), 0, np.sin(angle)], [
                              0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        return result

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        result = SO3()
        result.rot = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        return result

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        theta = 2*np.arccos(q[3])
        axis = q[:3]/np.linalg.norm(q[:3])
        v = theta * axis
        return SO3.exp(v)

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        q = np.zeros(4)
        theta = np.linalg.norm(self.log())
        q[3] = np.cos(theta/2)
        q[0:3] = np.sin(theta/2) * self.log()/theta
        return q

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        result = SO3()
        result.rot = np.eye(3) + np.sin(angle) * SO3.skew(axis) + \
            (1 - np.cos(angle)) * (SO3.skew(axis) @ SO3.skew(axis))
        return result

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        v = self.log()
        theta = np.linalg.norm(v)
        if theta == 0:
            return 0, np.array([0, 0, 1])
        axis = v/theta
        return theta, axis

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        result = SO3()
        letter_to_function = {'x': SO3.rx, 'y': SO3.ry, 'z': SO3.rz}
        for i, letter in enumerate(seq):
            result.rot = result.rot @ letter_to_function[letter](angles[i]).rot
        return result

    def __hash__(self):
        return id(self)
