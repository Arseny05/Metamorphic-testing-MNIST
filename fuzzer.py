import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class Fuzzer:

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()


    @staticmethod
    def transform_noise(batch, sigma):
        if sigma <= 0:
            return batch
        noice = torch.randn_like(batch) * sigma
        return torch.clamp(batch + noice, 0.0, 1.0)

    @staticmethod
    def transform_shift(batch, shift):

        if isinstance(shift, (int, float)):
            dx, dy = int(shift), 0

        else:
            dx, dy = int(shift[0]), int(shift[1])

        if dx == 0 and dy == 0:
            return batch

        return TF.affine(
            batch,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=0.0
        )

    @staticmethod
    def transform_rotate(batch, angle):
        if angle == 0:
            return batch
        else:
            return TF.rotate(batch, angle)

    @staticmethod
    def transform_scaling(batch, scale):
        if scale == 1.0 or scale == 0.0:
            return batch
        else:
            return TF.affine(
                batch,
                angle=0.0,
                translate = [0, 0],
                scale = float(scale),
                shear=[0.0, 0.0],
                fill=0.0
            )

    @staticmethod
    def transform_blur(batch, kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return batch
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        return TF.gaussian_blur(batch, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def evaluate_transformation(self, data_loader, transform_fn, param_val):
        total_samples = 0
        total_correct_clean = 0
        total_correct_transformed = 0
        violations_count = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                clean_logits = self.model(images)
                clean_preds = clean_logits.argmax(dim=1)
                clean_correct_mask = (clean_preds == labels)

                transformed_images = transform_fn(images, param_val)
                trans_logits = self.model(transformed_images)
                trans_preds = trans_logits.argmax(dim=1)
                trans_correct_mask = (trans_preds == labels)

                batch_size = labels.size(0)
                total_samples += batch_size
                total_correct_clean += clean_correct_mask.sum().item()
                total_correct_transformed += trans_correct_mask.sum().item()

                violations = clean_correct_mask & (~trans_correct_mask)
                violations_count += violations.sum().item()

        accuracy = total_correct_transformed / total_samples if total_samples > 0 else 0.0
        violation_rate = violations_count / total_correct_clean if total_correct_clean > 0 else 0.0

        return {
            "param": param_val,
            "accuracy": accuracy,
            "violation_rate": violation_rate
        }

    def run_parameter_grid(self, data_loader, transform_fn, param_grid):
        results = {"params": [], "accuracies": [], "violation_rates": []}
        for val in param_grid:
            res = self.evaluate_transformation(data_loader, transform_fn, val)
            results["params"].append(res["param"])
            results["accuracies"].append(res["accuracy"])
            results["violation_rates"].append(res["violation_rate"])
        return results

    def find_top_violations(self, data_loader, transform_fn, param_val, top_k=10):
        hard_violations = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                clean_logits = self.model(images)
                clean_probs = F.softmax(clean_logits, dim=1)
                clean_preds = clean_logits.argmax(dim=1)

                transformed = transform_fn(images, param_val)
                trans_logits = self.model(transformed)
                trans_probs = F.softmax(trans_logits, dim=1)
                trans_preds = trans_logits.argmax(dim=1)

                mask = (clean_preds == labels) & (trans_preds != labels)
                indices = torch.where(mask)[0]

                for idx in indices:
                    c_prob = clean_probs[idx, clean_preds[idx]].item()
                    t_prob = trans_probs[idx, trans_preds[idx]].item()
                    score = c_prob * t_prob
                    hard_violations.append({
                        "clean_img": images[idx].cpu(),
                        "trans_img": transformed[idx].cpu(),
                        "label": labels[idx].item(),
                        "clean_pred": clean_preds[idx].item(),
                        "clean_prob": c_prob,
                        "trans_pred": trans_preds[idx].item(),
                        "trans_prob": t_prob,
                        "score": score
                    })
        hard_violations.sort(key=lambda x: x["score"], reverse=True)
        return hard_violations[:top_k]

    