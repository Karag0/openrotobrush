{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: [
    ps.numpy
    ps.pillow
    ps.opencv4
    ps.ultralytics
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
    pkgs.tcl
    pkgs.xorg.libX11
  ];

  shellHook = ''
    export PATH="${pythonEnv}/bin:$PATH"
  '';
}