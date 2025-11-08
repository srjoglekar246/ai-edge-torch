# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A suite of tests to validate Mamba Cache layer."""

from ai_edge_torch.generative.layers import mamba_cache
import ai_edge_torch.generative.layers.model_config as cfg
import torch
import torch.utils._pytree as pytree

from absl.testing import absltest as googletest


class TestMambaLayers(googletest.TestCase):
    def _get_test_config(
        self,
        num_layers,
        hidden_size=768,
        chunk_size=256,
        d_conv=4,
        d_state=128,
        n_heads=48,
        d_head=32,
        n_groups=1,
        expand=2,
    ):
        mamba_config = cfg.MambaConfig(
            hidden_size=hidden_size,
            chunk_size=chunk_size,
            d_conv=d_conv,
            d_head=d_head,
            d_state=d_state,
            expand=expand,
            n_groups=n_groups,
            n_heads=n_heads,
        )
        block_config = cfg.TransformerBlockConfig(
            attn_config=None, ff_config=None, mamba_config=mamba_config
        )
        config = cfg.ModelConfig(
            embedding_dim=hidden_size,
            block_configs=block_config,
            num_layers=num_layers,
            max_seq_len=None,
            vocab_size=None,
        )
        return config

    def _assert_mamba_cache_entry_equal(self, cache1, cache2):
        self.assertIsInstance(cache1, mamba_cache.MambaCacheEntry)
        self.assertIsInstance(cache2, mamba_cache.MambaCacheEntry)
        self.assertTrue(torch.equal(cache1.conv_state, cache2.conv_state))
        self.assertTrue(torch.equal(cache1.ssm_state, cache2.ssm_state))

    def _assert_mamba_cache_equal(self, cache1, cache2):
        self.assertIsInstance(cache1, mamba_cache.MambaCache)
        self.assertIsInstance(cache2, mamba_cache.MambaCache)
        self.assertEqual(len(cache1.caches), len(cache2.caches))
        for cache1_entry, cache2_entry in zip(cache1.caches, cache2.caches):
            self._assert_mamba_cache_entry_equal(cache1_entry, cache2_entry)

    def test_cache_update(self):
        N = 1
        HIDDEN_SIZE = 768
        config = self._get_test_config(num_layers=N, hidden_size=HIDDEN_SIZE)
        cache = mamba_cache.MambaCache.from_model_config(config)
        entry = cache.caches[0]

        # Full update test
        new_conv_state = torch.full_like(entry.conv_state, 5.0)
        new_ssm_state = torch.full_like(entry.ssm_state, 7.0)

        updated_entry = mamba_cache.update(entry, new_conv_state, new_ssm_state)
        self.assertTrue(torch.equal(updated_entry.conv_state, new_conv_state))
        self.assertTrue(torch.equal(updated_entry.ssm_state, new_ssm_state))

    def test_serialization(self):
        class TestModel(torch.nn.Module):
            def forward(self, cache: mamba_cache.MambaCache) -> mamba_cache.MambaCache:
                updated_cache_entries = [
                    mamba_cache.MambaCacheEntry(
                        torch.zeros_like(entry.conv_state),
                        torch.zeros_like(entry.ssm_state),
                    )
                    for entry in cache.caches
                ]
                return mamba_cache.MambaCache(tuple(updated_cache_entries))

        N = 1
        config = self._get_test_config(num_layers=N)
        cache = mamba_cache.MambaCache.from_model_config(config)
        model = TestModel()
        exported_program = torch.export.export(model, (cache,))
        input_specs = exported_program.graph_signature.input_specs
        self.assertEqual(len(input_specs), 2)
        self.assertEqual(input_specs[0].arg.name, "cache_conv_state_0")
        self.assertEqual(input_specs[1].arg.name, "cache_ssm_state_0")

    def test_pytree_roundtrip_mamba_cache(self):
        NUM_LAYERS = 4
        config = self._get_test_config(num_layers=NUM_LAYERS)
        cache = mamba_cache.MambaCache.from_model_config(config, batch_size=1)
        flat, treespec = pytree.tree_flatten(cache)
        self.assertLen(flat, NUM_LAYERS * 2)
        cache_unflat = pytree.tree_unflatten(flat, treespec)
        self._assert_mamba_cache_equal(cache, cache_unflat)

    def test_pytree_roundtrip_mamba_entry(self):
        mamba_config = cfg.MambaConfig(
            hidden_size=768,
            chunk_size=256,
            d_conv=4,
            d_head=32,
            d_state=128,
            expand=2,
            n_groups=1,
            n_heads=48,
        )
        cache = mamba_cache.MambaCacheEntry.from_model_config(mamba_config)
        flat, treespec = pytree.tree_flatten(cache)
        self.assertLen(flat, 2)
        cache_unflat = pytree.tree_unflatten(flat, treespec)
        self._assert_mamba_cache_entry_equal(cache, cache_unflat)


if __name__ == "__main__":
    googletest.main()
