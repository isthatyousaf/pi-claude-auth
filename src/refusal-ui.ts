import {
	DynamicBorder,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	Container,
	type SelectItem,
	SelectList,
	Text,
} from "@earendil-works/pi-tui";

export type RefusalAction = "continue" | "edit";

export async function showRefusalMenu(
	ctx: ExtensionContext,
	refusedModelName: string,
	fallbackModelName: string,
	canEdit: boolean,
): Promise<RefusalAction | undefined> {
	const items: SelectItem[] = [
		{
			value: "continue",
			label: `Continue with ${fallbackModelName}`,
			description: "Keep completed work and continue from the current state",
		},
	];
	if (canEdit) {
		items.push({
			value: "edit",
			label: `Edit and retry with ${refusedModelName}`,
			description: "Branch before the refusal and restore the editor",
		});
	}

	return ctx.ui.custom<RefusalAction | undefined>(
		(tui, theme, _keybindings, done) => {
			const container = new Container();
			container.addChild(
				new DynamicBorder((text: string) => theme.fg("warning", text)),
			);
			container.addChild(
				new Text(theme.bold(theme.fg("warning", "Session paused")), 1, 0),
			);
			container.addChild(
				new Text(
					theme.fg(
						"muted",
						`${refusedModelName}'s safeguards flagged this response.`,
					),
					1,
					1,
				),
			);

			const selectList = new SelectList(items, items.length, {
				selectedPrefix: (text) => theme.fg("accent", text),
				selectedText: (text) => theme.fg("accent", text),
				description: (text) => theme.fg("muted", text),
				scrollInfo: (text) => theme.fg("dim", text),
				noMatch: (text) => theme.fg("warning", text),
			});
			selectList.onSelect = (item) => done(item.value as RefusalAction);
			selectList.onCancel = () => done(undefined);
			container.addChild(selectList);
			container.addChild(
				new Text(
					theme.fg("dim", "↑↓ navigate • enter select • esc stop"),
					1,
					0,
				),
			);
			container.addChild(
				new DynamicBorder((text: string) => theme.fg("warning", text)),
			);

			return {
				render: (width: number) => container.render(width),
				invalidate: () => container.invalidate(),
				handleInput: (data: string) => {
					selectList.handleInput(data);
					tui.requestRender();
				},
			};
		},
	);
}
