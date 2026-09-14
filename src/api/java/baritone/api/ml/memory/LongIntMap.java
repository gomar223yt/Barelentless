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

package baritone.api.ml.memory;

import java.util.Arrays;

/**
 * A minimal open-addressing map from {@code long} keys to {@code int} values.
 * <p>
 * Exists because the memory's hash tables are on the hot path of every lookup, and {@code HashMap<Long, ...>} boxes
 * both the key and every value it stores. At two hundred thousand records that boxing, and the pointer chasing that
 * comes with it, was most of the query time. This holds two primitive arrays and touches one or two cache lines per
 * probe.
 * <p>
 * Deliberately not a general purpose collection: it supports exactly the operations the memory needs, and treats
 * {@link #EMPTY} as the free slot marker, so that one key value cannot be stored. The memory's hashes are derived
 * from sign patterns and never take that value.
 *
 * @author Barelentless
 */
public final class LongIntMap {

    /**
     * The key value used to mark a free slot, and therefore the one key this map cannot hold.
     */
    public static final long EMPTY = Long.MIN_VALUE;

    /**
     * Returned when a key is absent.
     */
    public static final int ABSENT = -1;

    private long[] keys;
    private int[] values;
    private int size;
    private int mask;

    public LongIntMap(int expected) {
        int capacity = Integer.highestOneBit(Math.max(16, expected * 2 - 1)) * 2;
        this.keys = new long[capacity];
        this.values = new int[capacity];
        this.mask = capacity - 1;
        Arrays.fill(this.keys, EMPTY);
    }

    private static int mix(long key) {
        // fibonacci-style mixing; the incoming hashes are structured (sign patterns), so the low bits alone would
        // cluster badly under a plain mask
        long h = key * 0x9E3779B97F4A7C15L;
        h ^= h >>> 32;
        return (int) h;
    }

    public int get(long key) {
        int index = mix(key) & this.mask;
        while (true) {
            long current = this.keys[index];
            if (current == EMPTY) {
                return ABSENT;
            }
            if (current == key) {
                return this.values[index];
            }
            index = (index + 1) & this.mask;
        }
    }

    public void put(long key, int value) {
        if (key == EMPTY) {
            throw new IllegalArgumentException("cannot store the sentinel key");
        }
        int index = mix(key) & this.mask;
        while (true) {
            long current = this.keys[index];
            if (current == EMPTY) {
                this.keys[index] = key;
                this.values[index] = value;
                this.size++;
                if (this.size * 2 > this.keys.length) {
                    grow();
                }
                return;
            }
            if (current == key) {
                this.values[index] = value;
                return;
            }
            index = (index + 1) & this.mask;
        }
    }

    private void grow() {
        long[] oldKeys = this.keys;
        int[] oldValues = this.values;
        this.keys = new long[oldKeys.length * 2];
        this.values = new int[oldValues.length * 2];
        this.mask = this.keys.length - 1;
        Arrays.fill(this.keys, EMPTY);
        this.size = 0;
        for (int i = 0; i < oldKeys.length; i++) {
            if (oldKeys[i] != EMPTY) {
                put(oldKeys[i], oldValues[i]);
            }
        }
    }

    public void clear() {
        Arrays.fill(this.keys, EMPTY);
        this.size = 0;
    }

    public int size() {
        return this.size;
    }
}
