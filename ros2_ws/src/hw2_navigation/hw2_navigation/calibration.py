"""Inspect OpenCV calibration; check on held-out checkerboard images, or recalibrate."""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--images', help='quoted glob of RAW checkerboard images')
    parser.add_argument('--cols', type=int, default=8, help='interior corners, not squares')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--square', type=float, default=.025, help='MEASURE square edge, metres')
    parser.add_argument('--recalibrate-output', type=Path, help='write a NEW OpenCV YAML')
    args = parser.parse_args()
    fs = cv2.FileStorage(str(args.calibration), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        parser.error('cannot open OpenCV calibration YAML')
    k, d = fs.getNode('camera_matrix').mat(), fs.getNode('distortion_coefficients').mat()
    width, height = int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real())
    fs.release()
    if (k is None or d is None or k.shape != (3, 3) or not np.isfinite(k).all() or
            not np.isfinite(d).all() or min(width, height, k[0, 0], k[1, 1]) <= 0):
        parser.error('missing, nonfinite, or invalid intrinsic values')
    warnings = []
    if d.size >= 4 and max(abs(d.ravel()[2:4])) > .1:
        warnings.append('Large tangential distortion: verify coefficient order and held-out images')
    if not (0 <= k[0, 2] < width and 0 <= k[1, 2] < height):
        warnings.append('Principal point lies outside image')
    report = dict(resolution=[width, height], K=k.tolist(), D=d.ravel().tolist(), warnings=warnings,
                  note='Structure checks cannot certify geometric accuracy')
    if args.images:
        if min(args.cols, args.rows) < 2 or args.square <= 0:
            parser.error('invalid board dimensions')
        obj = np.zeros((args.cols*args.rows, 3), np.float32)
        obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)*args.square
        objects, corners, names = [], [], []
        for filename in sorted(glob.glob(args.images)):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape[::-1] != (width, height):
                parser.error(f'Unreadable or wrong-resolution image: {filename}')
            ok, pts = cv2.findChessboardCornersSB(img, (args.cols, args.rows))
            if ok:
                objects.append(obj.copy())
                corners.append(pts.astype(np.float32))
                names.append(filename)
        if not names:
            parser.error('no checkerboards found')
        if args.recalibrate_output:
            if len(names) < 15:
                parser.error('capture at least 15 varied board views before recalibration')
            if args.recalibrate_output.exists():
                parser.error('output already exists; choose a new file')
            rms, k, d, _, _ = cv2.calibrateCamera(objects, corners, (width, height), None, None)
            out = cv2.FileStorage(str(args.recalibrate_output), cv2.FILE_STORAGE_WRITE)
            for key, value in [('image_width', width), ('image_height', height), ('camera_name', 'camera'),
                               ('distortion_model', 'plumb_bob'), ('camera_matrix', k),
                               ('distortion_coefficients', d), ('rectification_matrix', np.eye(3)),
                               ('projection_matrix', np.column_stack([k, np.zeros(3)]))]:
                out.write(key, value)
            out.release()
            report['training_rms_px'] = rms
            report['note'] = 'Training fit only: rerun on separate held-out images using the new YAML'
        errors = []
        for filename, objpoints, pts in zip(names, objects, corners):
            ok, r, t = cv2.solvePnP(objpoints, pts, k, d)
            if not ok:
                continue
            projected, _ = cv2.projectPoints(objpoints, r, t, k, d)
            errors.append(dict(image=filename,
                rms_px=float(np.sqrt(np.mean(np.sum((projected-pts)**2, axis=2))))))
        report['views'] = errors
    elif args.recalibrate_output:
        parser.error('--recalibrate-output requires --images')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
