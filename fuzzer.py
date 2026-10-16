import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Fuzzer:

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def transform_noise(batch, sigma):
        if sigma <= 0:
            return batch
        noise = torch.randn_like(batch) * sigma
        return torch.clamp(batch + noise, 0.0, 1.0)

    @staticmethod
    def transform_shift(batch, shift):
        if isinstance(shift, (int, float)):
            dx, dy = int(shift), 0
        else:
            dx, dy = int(shift[0]), int(shift[1])

        if dx == 0 and dy == 0:
            return batch

        transformed = TF.affine(
            batch,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=0.0,
        )
        return torch.clamp(transformed, 0.0, 1.0)

    @staticmethod
    def transform_rotate(batch, angle):
        if angle == 0:
            return batch
        transformed = TF.rotate(batch, angle, fill=0.0)
        return torch.clamp(transformed, 0.0, 1.0)

    @staticmethod
    def transform_scaling(batch, scale):
        if scale == 1.0 or scale == 0.0:
            return batch
        transformed = TF.affine(
            batch,
            angle=0.0,
            translate=[0, 0],
            scale=float(scale),
            shear=[0.0, 0.0],
            fill=0.0,
        )
        return torch.clamp(transformed, 0.0, 1.0)

    @staticmethod
    def transform_blur(batch, kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return batch
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        transformed = TF.gaussian_blur(
            batch, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        )
        return torch.clamp(transformed, 0.0, 1.0)

    def evaluate_transformation(self, data_loader, transform_fn, param_val):
        
        total_samples = 0
        violations_count = 0
        total_correct_transformed = 0
        has_labels = False

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    has_labels = True
                else:
                    images = batch.to(self.device)
                    labels = None

                batch_size = images.size(0)
                total_samples += batch_size

                clean_logits = self.model(images)
                clean_preds = clean_logits.argmax(dim=1)

                transformed_images = transform_fn(images, param_val)
                trans_logits = self.model(transformed_images)
                trans_preds = trans_logits.argmax(dim=1)

                violations = clean_preds != trans_preds
                violations_count += violations.sum().item()

                if has_labels:
                    total_correct_transformed += (trans_preds == labels).sum().item()

        violation_rate = (
            violations_count / total_samples if total_samples > 0 else 0.0
        )
        accuracy = (
            total_correct_transformed / total_samples
            if (has_labels and total_samples > 0)
            else None
        )

        return {
            "param": param_val,
            "violation_rate": violation_rate,
            "accuracy": accuracy,
        }

    def run_parameter_grid(self, data_loader, transform_fn, param_grid):
        results = {"params": [], "violation_rates": [], "accuracies": []}
        for val in param_grid:
            res = self.evaluate_transformation(data_loader, transform_fn, val)
            results["params"].append(res["param"])
            results["violation_rates"].append(res["violation_rate"])
            results["accuracies"].append(res["accuracy"])
        return results

    def find_top_violations(self, data_loader, transform_fn, param_val, top_k=10):
        
        hard_violations = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    images = batch.to(self.device)
                    labels = None

                clean_logits = self.model(images)
                clean_probs = F.softmax(clean_logits, dim=1)
                clean_preds = clean_logits.argmax(dim=1)

                transformed = transform_fn(images, param_val)
                trans_logits = self.model(transformed)
                trans_probs = F.softmax(trans_logits, dim=1)
                trans_preds = trans_logits.argmax(dim=1)

                violation_mask = clean_preds != trans_preds
                indices = torch.where(violation_mask)[0]

                for idx in indices:
                    c_pred = clean_preds[idx]
                    t_pred = trans_preds[idx]
                    c_prob = clean_probs[idx, c_pred].item()
                    t_prob = trans_probs[idx, t_pred].item()

                    score = c_prob * t_prob

                    hard_violations.append({
                        "clean_img": images[idx].cpu(),
                        "trans_img": transformed[idx].cpu(),
                        "label": labels[idx].item() if labels is not None else None,
                        "clean_pred": c_pred.item(),
                        "clean_prob": c_prob,
                        "trans_pred": t_pred.item(),
                        "trans_prob": t_prob,
                        "score": score,
                    })

        hard_violations.sort(key=lambda x: x["score"], reverse=True)
        return hard_violations[:top_k]

    