function gen_arch_config() {
    const config = {};

    /* swin_unet */
    config["swin_unet"] = {
        art: { color_stability: true, padding: "replication" },
        art_scan: { color_stability: false, padding: "replication" },
        photo: { color_stability: false, padding: "reflection" }
    };
    const swin = config["swin_unet"];
    const calc_tile_size_swin_unet = function (tile_size) {
        while (true) {
            if ((tile_size - 16) % 12 == 0 && (tile_size - 16) % 16 == 0) {
                break;
            }
            tile_size += 1;
        }
        return tile_size;
    };
    for (const domain of ["art", "art_scan", "photo"]) {
        const base_config = {
            ...swin[domain],
            arch: "swin_unet",
            domain: domain,
            calc_tile_size: calc_tile_size_swin_unet
        };
        swin[domain] = {
            scale2x: { ...base_config, scale: 2, offset: 16 },
            scale4x: { ...base_config, scale: 4, offset: 32 },
            scale1x: { ...base_config, scale: 1, offset: 8 }, // bypass for alpha denoise
        };
        for (let i = 0; i < 4; ++i) {
            swin[domain]["noise" + i + "_scale2x"] = { ...base_config, scale: 2, offset: 16 };
            swin[domain]["noise" + i + "_scale4x"] = { ...base_config, scale: 4, offset: 32 };
            swin[domain]["noise" + i] = { ...base_config, scale: 1, offset: 8 };
        }
    }
    /* cunet */
    config["cunet"] = { art: {} };
    const calc_tile_size_cunet = function (tile_size, config) {
        const adj = config.scale == 1 ? 16 : 32;
        tile_size = ((tile_size * config.scale + config.offset * 2) - adj) / config.scale;
        tile_size -= tile_size % 4;
        return tile_size;
    };
    const base_config = {
        arch: "cunet",
        domain: "art",
        calc_tile_size: calc_tile_size_cunet,
        color_stability: true,
        padding: "replication",
    };
    config["cunet"]["art"] = {
        scale2x: { ...base_config, scale: 2, offset: 36 },
        scale1x: { ...base_config, scale: 1, offset: 28 }, // bypass for alpha denoise
    };
    const base = config["cunet"];
    for (let i = 0; i < 4; ++i) {
        base["art"]["noise" + i + "_scale2x"] = { ...base_config, scale: 2, offset: 36 };
        base["art"]["noise" + i] = { ...base_config, scale: 1, offset: 28 };
    }

    return config;
}

export const CONFIG = {
    arch: gen_arch_config(),
    get_config: function (arch, style, method) {
        if ((arch in this.arch) && (style in this.arch[arch]) && (method in this.arch[arch][style])) {
            const config = { ...this.arch[arch][style][method] };
            config["path"] = `models/${arch}/${style}/${method}.onnx`;
            return config;
        } else {
            return null;
        }
    },
    get_helper_model_path: function (name) {
        return `models/utils/${name}.onnx`;
    }
};
