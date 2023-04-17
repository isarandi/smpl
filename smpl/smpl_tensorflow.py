import os
import pickle

import numpy as np
import tensorflow as tf


class SMPL:
    def __init__(self, model_root, gender='neutral'):
        """
        Args:
            model_root: path to pickle files for the model (see https://smpl.is.tue.mpg.de).
            gender: 'neutral', 'f' or 'm'
        """
        super().__init__()
        self.gender = gender
        if gender[0] == 'f':
            model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        elif gender[0] == 'm':
            model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        else:
            model_path = os.path.join(model_root, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

        with open(model_path, 'rb') as f:
            self.smpl_data = pickle.load(f, encoding='latin1')

        self.shapedirs = tf.constant(np.array(self.smpl_data['shapedirs']), dtype=tf.float32)
        self.posedirs = tf.constant(np.array(self.smpl_data['posedirs']), dtype=tf.float32)
        self.v_template = tf.constant(np.expand_dims(self.smpl_data['v_template'], axis=0),
                                      dtype=tf.float32)
        self.J_regressor = tf.constant(
            np.array(self.smpl_data['J_regressor'].toarray()), dtype=tf.float32)
        self.weights = tf.constant(np.array(self.smpl_data['weights']), dtype=tf.float32)
        self.faces = tf.constant(np.array(self.smpl_data['f'].astype(np.int32)))

        self.kintree_parents = self.smpl_data['kintree_table'][0].tolist()
        self.num_joints = len(self.kintree_parents)

        self.J_shapedirs = tf.einsum('jv,vcs->jcs', self.J_regressor, self.shapedirs)
        self.J_template = self.J_regressor @ self.v_template
        self.v_dirs = tf.concat([self.shapedirs, self.posedirs], axis=2)
        self.v_template = self.v_template - tf.einsum('vcx,x->vc', self.posedirs, tf.reshape(
            tf.tile(tf.eye(3, dtype=tf.float32), [self.num_joints - 1, 1]), [-1]))

    def __call__(self, pose_rotvecs, shape_betas, trans=None, return_vertices=True):
        """Calculate the SMPL body model vertices, joint positions and orientations given the input
         pose and shape parameters.

        Args:
            pose_rotvecs (tf.Tensor): A tensor of shape (batch_size, num_joints * 3),
                representing the rotation vectors for each joint in the pose.
            shape_betas (tf.Tensor): A tensor of shape (batch_size, num_shape_coeffs),
                representing the shape coefficients (betas) for the body shape.
            trans (tf.Tensor, optional): A tensor of shape (batch_size, 3), representing the
                translation of the root joint. Defaults to None, in which case a zero translation is
                applied.
            return_vertices (bool, optional): A flag indicating whether to return the body model
                vertices. If False, only joint positions and orientations are returned.
                Defaults to True.

        Returns:
            A dictionary with following keys and values:
                - 'vertices': A tensor of shape (batch_size, num_vertices, 3), representing the
                    3D body model vertices in the posed state. This key is only present if
                    `return_vertices` is True.
                - 'joints': A tensor of shape (batch_size, num_joints, 3), representing the 3D
                    positions of the body joints.
                - 'orientations': A tensor of shape (batch_size, num_joints, 3, 3), representing
                    the 3D orientation matrices for each joint.
        """
        pose_rotvecs = tf.cast(pose_rotvecs, tf.float32)
        shape_betas = tf.cast(shape_betas, tf.float32)
        if trans is None:
            trans = tf.zeros((1, 3), tf.float32)
        trans = tf.cast(trans, tf.float32)

        j = self.J_template + tf.einsum('jcs,bs->bjc', self.J_shapedirs, shape_betas)

        rel_rotmats = rotvec2mat(tf.reshape(pose_rotvecs, (-1, self.num_joints, 3)))
        glob_rotmats = [rel_rotmats[:, 0]]
        glob_positions = [j[:, 0] + trans]

        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_positions.append(
                glob_positions[i_parent] +
                tf.einsum('bCc,bc->bC', glob_rotmats[i_parent], j[:, i_joint] - j[:, i_parent]))

        glob_rotmats = tf.stack(glob_rotmats, axis=1)
        glob_positions = tf.stack(glob_positions, axis=1)

        if not return_vertices:
            return dict(joints=glob_positions, orientations=glob_rotmats)

        params = tf.concat([
            shape_betas,
            tf.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])], axis=1)
        v_posed = self.v_template + tf.einsum('vcp,bp->bvc', self.v_dirs, params)
        translations = glob_positions - tf.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
                tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed) +
                self.weights @ translations)

        return dict(vertices=vertices, joints=glob_positions, orientations=glob_rotmats)


def rotvec2mat(rotvec):
    angle = tf.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = tf.math.divide_no_nan(rotvec, angle)

    sin_axis = tf.sin(angle) * axis
    cos_angle = tf.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
    matrix = tf.stack((diag_x, m01, m02,
                       m10, diag_y, m12,
                       m20, m21, diag_z), axis=-1)
    return tf.reshape(matrix, tf.concat((tf.shape(axis)[:-1], (3, 3)), axis=-1))
