/*
 * This file is part of Baritone.
 *
 * Baritone is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Baritone is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Baritone.  If not, see <https://www.gnu.org/licenses/>.
 */

package baritone.api.ml.io;

import baritone.api.ml.Tensor;
import baritone.api.ml.nn.Module;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Checkpoint format for trained models.
 * <p>
 * Parameters are stored by qualified name rather than by position, so a checkpoint keeps loading after layers are
 * added, renamed or reordered - anything that no longer matches is reported instead of silently corrupting the model.
 * Saves are atomic (write to a temporary file, then move) because the trainer writes checkpoints while the game is
 * running and a crash mid-write must not cost the player their trained weights.
 *
 * @author Barelentless
 */
public final class ModelIO {

    private static final int MAGIC = 0x42524C4D; // "BRLM"
    private static final int VERSION = 1;

    private ModelIO() {}

    /**
     * The result of loading a checkpoint: how many parameters matched, plus the metadata that was stored with it.
     */
    public static final class LoadReport {

        public final int matched;
        public final int missing;
        public final int mismatched;
        public final Map<String, String> metadata;

        LoadReport(int matched, int missing, int mismatched, Map<String, String> metadata) {
            this.matched = matched;
            this.missing = missing;
            this.mismatched = mismatched;
            this.metadata = metadata;
        }

        public boolean isClean() {
            return this.missing == 0 && this.mismatched == 0;
        }

        @Override
        public String toString() {
            return "loaded " + this.matched + " tensors"
                    + (this.missing > 0 ? ", " + this.missing + " missing" : "")
                    + (this.mismatched > 0 ? ", " + this.mismatched + " shape mismatches" : "");
        }
    }

    public static void save(Module module, Path path, Map<String, String> metadata) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
        try (OutputStream raw = Files.newOutputStream(temporary);
             DataOutputStream out = new DataOutputStream(new GZIPOutputStream(new BufferedOutputStream(raw)))) {
            write(module, out, metadata);
        }
        Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
    }

    public static void write(Module module, DataOutputStream out, Map<String, String> metadata) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        Map<String, String> meta = metadata == null ? new LinkedHashMap<>() : metadata;
        out.writeInt(meta.size());
        for (Map.Entry<String, String> entry : meta.entrySet()) {
            out.writeUTF(entry.getKey());
            out.writeUTF(entry.getValue());
        }
        Map<String, Tensor> parameters = module.namedParameters();
        out.writeInt(parameters.size());
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            Tensor tensor = entry.getValue();
            out.writeUTF(entry.getKey());
            out.writeInt(tensor.rows);
            out.writeInt(tensor.cols);
            for (float value : tensor.data) {
                out.writeFloat(value);
            }
        }
    }

    public static LoadReport load(Module module, Path path) throws IOException {
        try (InputStream raw = Files.newInputStream(path);
             DataInputStream in = new DataInputStream(new GZIPInputStream(new BufferedInputStream(raw)))) {
            return read(module, in);
        }
    }

    public static LoadReport read(Module module, DataInputStream in) throws IOException {
        if (in.readInt() != MAGIC) {
            throw new IOException("not a Barelentless model file");
        }
        int version = in.readInt();
        if (version > VERSION) {
            throw new IOException("model was written by a newer version (" + version + " > " + VERSION + ")");
        }
        Map<String, String> metadata = new LinkedHashMap<>();
        int metaCount = in.readInt();
        for (int i = 0; i < metaCount; i++) {
            metadata.put(in.readUTF(), in.readUTF());
        }
        Map<String, Tensor> parameters = module.namedParameters();
        int count = in.readInt();
        int matched = 0;
        int mismatched = 0;
        for (int i = 0; i < count; i++) {
            String name = in.readUTF();
            int rows = in.readInt();
            int cols = in.readInt();
            Tensor target = parameters.remove(name);
            if (target == null || target.rows != rows || target.cols != cols) {
                if (target != null) {
                    mismatched++;
                }
                in.skipBytes(rows * cols * 4);
                continue;
            }
            for (int j = 0; j < target.data.length; j++) {
                target.data[j] = in.readFloat();
            }
            matched++;
        }
        return new LoadReport(matched, parameters.size(), mismatched, metadata);
    }

    /**
     * Reads only the metadata block, without touching the weights. Lets the UI describe a checkpoint (when it was
     * trained, on how many samples, by which architecture) before deciding whether to load it.
     */
    public static Map<String, String> peekMetadata(Path path) throws IOException {
        try (InputStream raw = Files.newInputStream(path);
             DataInputStream in = new DataInputStream(new GZIPInputStream(new BufferedInputStream(raw)))) {
            if (in.readInt() != MAGIC) {
                throw new IOException("not a Barelentless model file");
            }
            in.readInt();
            Map<String, String> metadata = new LinkedHashMap<>();
            int metaCount = in.readInt();
            for (int i = 0; i < metaCount; i++) {
                metadata.put(in.readUTF(), in.readUTF());
            }
            return metadata;
        }
    }
}
