#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run All Tests

全てのテストを実行
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_test_module(module_name):
    """テストモジュールを実行"""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print('='*60)

    try:
        if module_name == 'test_model':
            import test_model
            test_model.test_simple_sage_import()
            test_model.test_simple_sage_initialization()
            test_model.test_simple_sage_forward()
            test_model.test_simple_sage_different_sizes()
            test_model.test_simple_sage_gradient()
            print(f"\n✅ {module_name} - All tests passed!")

        elif module_name == 'test_utils':
            import test_utils
            test_utils.test_matvec_csr_import()
            test_utils.test_matvec_csr_simple()
            test_utils.test_matvec_csr_identity()
            test_utils.test_matvec_csr_zero()
            test_utils.test_matvec_csr_gradient()
            print(f"\n✅ {module_name} - All tests passed!")

        elif module_name == 'test_losses':
            import test_losses
            test_losses.test_mesh_quality_import()
            test_losses.test_get_reference_values()
            test_losses.test_get_weight_coefficients()
            test_losses.test_build_w_pde_basic()
            test_losses.test_build_w_pde_good_quality()
            test_losses.test_build_w_pde_poor_quality()
            test_losses.test_build_w_pde_dtype()
            test_losses.test_build_w_pde_custom_max()
            print(f"\n✅ {module_name} - All tests passed!")

        elif module_name == 'test_config':
            import test_config
            test_config.test_config_import()
            test_config.test_default_config()
            test_config.test_data_config_defaults()
            test_config.test_model_config_defaults()
            test_config.test_training_config_defaults()
            test_config.test_mesh_quality_config_defaults()
            test_config.test_create_custom_config()
            test_config.test_config_modification()
            print(f"\n✅ {module_name} - All tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ {module_name} - Test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全てのテストを実行"""
    print("="*60)
    print("GNNtest2 - Test Suite")
    print("="*60)

    test_modules = [
        'test_config',
        'test_model',
        'test_utils',
        'test_losses',
    ]

    results = {}
    for module in test_modules:
        results[module] = run_test_module(module)

    # 結果サマリー
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    total = len(test_modules)
    passed = sum(results.values())
    failed = total - passed

    for module, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{module:20s} : {status}")

    print("="*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("="*60)

    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed} test module(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
