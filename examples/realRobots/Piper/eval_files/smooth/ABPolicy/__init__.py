"""从 ABPolicy(third_party/ABPolicy-code)复制进来的平滑算法代码。

复制进包以去掉 third_party 路径依赖;逻辑未改。
- curve_fitter.py: BSplineFitter —— B-spline 拟合(fit/rebuild)+ 异步块焊接(refit_prefix/refit_prefix_w)

用法:  from smooth.ABPolicy import BSplineFitter        # 需 scipy
自检:  python -c "import sys; sys.path.insert(0,'deployment/piper'); from smooth.ABPolicy import demo; demo()"
"""

__all__ = ["BSplineFitter", "demo"]


def __getattr__(name):
    # 懒导出:只有真正取用 BSplineFitter 时才 import curve_fitter(从而触发 scipy);
    # 只 import 本子包不强制装 scipy。
    if name == "BSplineFitter":
        from .curve_fitter import BSplineFitter
        return BSplineFitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def demo():
    """自检复制来的 BSplineFitter:fit→rebuild 能还原平滑曲线;refit_prefix_w 把前缀焊向 past。需 scipy。"""
    import numpy as np

    from .curve_fitter import BSplineFitter

    T, D = 20, 3
    t = np.linspace(0, 1, T)[:, None]
    y = np.sin(2 * np.pi * t * np.arange(1, D + 1)).astype(np.float32)   # (T,D) 平滑曲线

    f = BSplineFitter(T=T, k=3, n_ctrl=8)
    ctrl = f.fit(y)
    y_hat, _ = f.rebuild(ctrl)
    assert y_hat.shape == y.shape, "rebuild 形状不符"

    # 假装前 8 步是"已执行 past"(整体偏移 0.1), refit_prefix_w 应把新块前缀焊过去
    past = (y[:8] + 0.1).astype(np.float32)
    new_ctrl = f.refit_prefix_w(past, ctrl, n_prefix=8, n_free=4, last_pt_weight=0.05)
    welded, _ = f.rebuild(new_ctrl)
    assert welded.shape == y.shape
    before = float(np.abs(y_hat[:8] - past).mean())
    after = float(np.abs(welded[:8] - past).mean())
    assert after < before, f"焊接未让前缀更贴近 past: before={before:.4f} after={after:.4f}"

    print(f"ABPolicy BSplineFitter demo OK (前缀对 past 误差 {before:.3f} -> {after:.3f})")


if __name__ == "__main__":
    demo()
