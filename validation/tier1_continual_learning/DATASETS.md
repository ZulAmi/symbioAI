# Continual Learning Benchmark Datasets

This document describes all datasets supported in the industry-standard continual learning benchmarks.

## 📊 Supported Datasets Overview

| Dataset           | Classes | Train Size | Test Size | Image Size | Channels | Status   |
| ----------------- | ------- | ---------- | --------- | ---------- | -------- | -------- |
| **MNIST**         | 10      | 60,000     | 10,000    | 28×28      | 1 (Gray) | ✅ Ready |
| **Fashion-MNIST** | 10      | 60,000     | 10,000    | 28×28      | 1 (Gray) | ✅ Ready |
| **CIFAR-10**      | 10      | 50,000     | 10,000    | 32×32      | 3 (RGB)  | ✅ Ready |
| **CIFAR-100**     | 100     | 50,000     | 10,000    | 32×32      | 3 (RGB)  | ✅ Ready |
| **SVHN**          | 10      | 73,257     | 26,032    | 32×32      | 3 (RGB)  | ✅ Ready |
| **Omniglot**      | 1,623   | 19,280     | 13,180    | 105×105    | 1 (Gray) | ✅ Ready |
| **TinyImageNet**  | 200     | 100,000    | 10,000    | 64×64      | 3 (RGB)  | ✅ Ready |

---

## 📝 Dataset Details

### MNIST

**Handwritten Digits**

- **Papers using it**: 10,000+ (foundational dataset)
- **Continual Learning Papers**: EWC, ER, GEM, A-GEM, PackNet
- **Use case**: Baseline, quick prototyping
- **Auto-download**: ✅ Yes (torchvision)

```python
python industry_standard_benchmarks.py --dataset mnist --strategy optimized
```

---

### Fashion-MNIST

**Fashion Product Images**

- **Papers using it**: 500+ papers
- **Continual Learning Papers**: iCaRL, LwF, SI, MAS
- **Use case**: More challenging than MNIST, same dimensions
- **Auto-download**: ✅ Yes (torchvision)
- **Classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

```python
python industry_standard_benchmarks.py --dataset fashion_mnist --strategy optimized
```

---

### CIFAR-10

**Natural Images (10 Classes)**

- **Papers using it**: 5,000+ papers
- **Continual Learning Papers**: All major CL papers use this
- **Use case**: Standard RGB benchmark
- **Auto-download**: ✅ Yes (torchvision)
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

```python
python industry_standard_benchmarks.py --dataset cifar10 --strategy optimized
```

**Why it's important**:

- Standard augmentation from ResNet paper (He et al. 2016)
- Used in every continual learning paper for comparison
- RGB color images (more realistic than grayscale)

---

### CIFAR-100

**Natural Images (100 Classes)**

- **Papers using it**: 2,000+ papers
- **Continual Learning Papers**: DER++, Co2L, REMIND, ER-Ring, Dark Experience Replay
- **Use case**: Fine-grained classification, more tasks
- **Auto-download**: ✅ Yes (torchvision)
- **Classes**: 100 classes in 20 superclasses

```python
python industry_standard_benchmarks.py --dataset cifar100 --strategy optimized
```

**Why it's important**:

- **Most popular continual learning benchmark** (100+ recent papers)
- 10× more classes = better test of catastrophic forgetting
- Standard split: 5-10 tasks with 10-20 classes each
- Essential for competitive benchmarks

---

### SVHN (Street View House Numbers)

**Real-world Digit Recognition**

- **Papers using it**: 1,000+ papers
- **Continual Learning Papers**: Domain-IL benchmarks, RODEO, RtF
- **Use case**: Domain adaptation, cross-domain transfer
- **Auto-download**: ✅ Yes (torchvision)
- **Source**: Google Street View images

```python
python industry_standard_benchmarks.py --dataset svhn --strategy optimized
```

**Why it's important**:

- Real-world difficulty (lighting, angles, backgrounds)
- Standard for domain adaptation benchmarks
- Tests transfer from MNIST → SVHN (different domain, same task)

---

### Omniglot

**Handwritten Characters (1,623 Classes)**

- **Papers using it**: 500+ papers (meta-learning, few-shot)
- **Continual Learning Papers**: MAML, Reptile, OML, Meta-SGD
- **Use case**: Few-shot learning, meta-learning benchmarks
- **Auto-download**: ✅ Yes (torchvision)
- **Source**: 50 alphabets, 20 examples per character

```python
python industry_standard_benchmarks.py --dataset omniglot --strategy optimized
```

**Why it's important**:

- Standard for few-shot and meta-learning
- Huge number of classes (1,623) with few examples
- Tests learning from limited data
- Complements your one-shot meta-learning capabilities

---

### TinyImageNet-200 ⭐ PRIORITY

**ImageNet Subset (200 Classes)**

- **Papers using it**: **100+ continual learning papers**
- **Continual Learning Papers**: DER++, Co2L, REMIND, ER-Ring, Rainbow Memory, GDumb
- **Use case**: **Production-scale benchmarking**
- **Manual download**: Required (see below)
- **Classes**: 200 ImageNet classes, 500 train + 50 val per class

```python
python industry_standard_benchmarks.py --dataset tiny_imagenet --strategy optimized
```

**Why it's CRITICAL**:

- ✅ **Most cited continual learning benchmark** in recent papers (2020-2025)
- ✅ Used by **every competitive startup** and research lab
- ✅ Larger images (64×64) = more realistic
- ✅ More classes (200) = better forgetting test
- ✅ **Required for credibility** when comparing to other AI startups

**Download & Setup**:

```bash
# Option 1: Use our script
./validation/tier1_continual_learning/download_tiny_imagenet.sh

# Option 2: Manual download
cd data
curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# Prepare validation set (required!)
python3 validation/tier1_continual_learning/prepare_tiny_imagenet.py
```

**Your Status**: ✅ Already downloaded and prepared!

---

## 🎯 Recommended Benchmark Strategy

### For Research Papers

```bash
# Start with small datasets
python industry_standard_benchmarks.py --dataset mnist
python industry_standard_benchmarks.py --dataset fashion_mnist

# Standard benchmarks
python industry_standard_benchmarks.py --dataset cifar10
python industry_standard_benchmarks.py --dataset cifar100  # Essential!

# Advanced benchmarks
python industry_standard_benchmarks.py --dataset tiny_imagenet  # Critical!
```

### For Competitive Analysis (vs AI Startups)

```bash
# Run these for credibility:
1. CIFAR-100 (most cited recent benchmark)
2. TinyImageNet-200 (production-scale)
3. SVHN (domain adaptation)

# These differentiate you:
4. Omniglot (few-shot learning)
```

---

## 📈 Dataset Statistics & Normalization

All datasets use literature-standard normalization:

```yaml
mnist:
  mean: [0.1307]
  std: [0.3081]

fashion_mnist:
  mean: [0.2860]
  std: [0.3530]

cifar10:
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2023, 0.1994, 0.2010]

cifar100:
  mean: [0.5071, 0.4867, 0.4408]
  std: [0.2675, 0.2565, 0.2761]

svhn:
  mean: [0.4377, 0.4438, 0.4728]
  std: [0.1980, 0.2010, 0.1970]

omniglot:
  mean: [0.0823]
  std: [0.2660]

tiny_imagenet:
  mean: [0.4802, 0.4481, 0.3975]
  std: [0.2302, 0.2265, 0.2262]
```

---

## 🔬 Data Augmentation

### Grayscale (MNIST, Fashion-MNIST, Omniglot)

- RandomRotation(10°)
- RandomAffine(translate=0.1)

### RGB Small (CIFAR-10, CIFAR-100, SVHN)

- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- ColorJitter(0.1, 0.1, 0.1)

### RGB Large (TinyImageNet)

- RandomCrop(64, padding=8)
- RandomHorizontalFlip()
- ColorJitter(0.2, 0.2, 0.2, 0.1)
- RandomRotation(15°)

---

## 🏆 Benchmark Comparisons

### Your Results vs Published Papers

When you run benchmarks, compare against:

**CIFAR-100 (10 tasks, 10 classes each)**:

- DER++: 76.2% final accuracy, 10.5% forgetting
- Co2L: 78.1% final accuracy, 8.3% forgetting
- ER-Ring: 73.5% final accuracy, 12.1% forgetting
- **Your goal**: 70%+ accuracy, <15% forgetting

**TinyImageNet-200 (10 tasks, 20 classes each)**:

- DER++: 62.4% final accuracy, 15.2% forgetting
- REMIND: 65.1% final accuracy, 12.8% forgetting
- **Your goal**: 60%+ accuracy, <20% forgetting

---

## 📚 References

- **MNIST**: LeCun et al. (1998) - Foundational dataset
- **Fashion-MNIST**: Xiao et al. (2017) - MNIST replacement
- **CIFAR-10/100**: Krizhevsky (2009) - Standard vision benchmark
- **SVHN**: Netzer et al. (2011) - Real-world digit recognition
- **Omniglot**: Lake et al. (2015) - Few-shot learning
- **TinyImageNet**: Stanford CS231n - ImageNet subset

### Continual Learning Papers Using These Datasets:

- Kirkpatrick et al. (2017) - EWC (MNIST, CIFAR)
- Rebuffi et al. (2017) - iCaRL (CIFAR-100)
- Buzzega et al. (2020) - DER++ (CIFAR-100, TinyImageNet)
- Cha et al. (2021) - Co2L (CIFAR-100, TinyImageNet)
- Hayes et al. (2020) - REMIND (TinyImageNet)

---

## 💡 Usage Tips

### Quick Test

```bash
# Fast iteration with MNIST (60 seconds)
python industry_standard_benchmarks.py --dataset mnist --strategy optimized
```

### Standard Benchmark

```bash
# Publication-ready with CIFAR-100 (30 minutes)
python industry_standard_benchmarks.py --dataset cifar100 --strategy optimized
```

### Production Validation

```bash
# Real-world scale with TinyImageNet (2-4 hours)
python industry_standard_benchmarks.py --dataset tiny_imagenet --strategy optimized
```

### All Strategies Comparison

```bash
for strategy in naive replay ewc multihead optimized; do
    python industry_standard_benchmarks.py --dataset cifar100 --strategy $strategy
done
```

---

## ✅ Checklist for Competitive Benchmarking

For competing with AI startups, you need:

- [x] MNIST (baseline)
- [x] Fashion-MNIST (baseline)
- [x] CIFAR-10 (standard)
- [x] CIFAR-100 (**essential** - most cited)
- [x] SVHN (domain adaptation)
- [x] Omniglot (few-shot/meta-learning)
- [x] TinyImageNet-200 (**critical** - production scale)

**Status**: ✅ **ALL DATASETS READY!**

You now have the complete benchmark suite used by leading continual learning research labs and AI startups.

---

## 🚀 Next Steps

1. **Run CIFAR-100 first** (most important for comparisons)
2. **Run TinyImageNet** (proves production readiness)
3. **Compare your results** against published papers (see above)
4. **Showcase unique features**: Neural-symbolic reasoning, causal meta-learning, multi-agent coordination

Your competitive advantage isn't just the benchmarks—it's the **unique capabilities** (explainability, causality, lifelong learning) that these benchmarks help demonstrate.
