{
  description = "Iris - C Image Generation Engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "iris";
          version = "0.1.0";
          src = ./.;

          buildInputs = [
            # For Linux BLAS support
            pkgs.openblas
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Accelerate
            pkgs.darwin.apple_sdk.frameworks.Metal
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShadersGraph
            pkgs.darwin.apple_sdk.frameworks.Foundation
          ];

          buildPhase = ''
            if [ "${system}" = "aarch64-darwin" ]; then
              make mps lib
            elif [ "${system}" = "x86_64-darwin" ] || [ "${if pkgs.stdenv.isLinux then "1" else "0"}" = "1" ]; then
              make blas lib
            else
              make generic lib
            fi
          '';

          installPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/lib
            mkdir -p $out/include
            cp iris $out/bin/
            cp libiris.a $out/lib/
            cp iris.h $out/include/
            cp iris_kernels.h $out/include/
          '';

          meta = with pkgs.lib; {
            description = "A C inference pipeline for image synthesis models";
            homepage = "https://github.com/lucidrains/iris";
            license = licenses.mit;
            platforms = platforms.all;
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.gnumake
            pkgs.gcc
            pkgs.openblas
            (pkgs.python3.withPackages (ps: [
              ps.huggingface-hub
              ps.numpy
              ps.pillow
            ]))
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Accelerate
            pkgs.darwin.apple_sdk.frameworks.Metal
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
            pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShadersGraph
            pkgs.darwin.apple_sdk.frameworks.Foundation
          ];
        };
      }
    );
}
